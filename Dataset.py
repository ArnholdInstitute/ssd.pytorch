from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
import glob, pdb, os, re, json, gdal, random, cv2, numpy as np, torch, boto3
from shapely.geometry import shape
from datetime import datetime
from skimage import io

def proj_to_raster(ds, projx, projy):
    gm = ds.GetGeoTransform()

    # Transform per inverse of http://www.gdal.org/gdal_datamodel.html
    x = (gm[5] * (projx - gm[0]) - gm[2] * (projy - gm[3])) / \
        (gm[5] * gm[1] + gm[4] * gm[2])
    y = (projy - gm[3] - x * gm[4]) / gm[5]
    return x, y

def raster_to_proj(ds, x, y):
    gm = ds.GetGeoTransform()

    # Transform per http://www.gdal.org/gdal_datamodel.html
    projx = gm[0] + gm[1] * x + gm[2] * y
    projy = gm[3] + gm[4] * x + gm[5] * y
    return projx, projy

class SatelliteDataset(data.Dataset):
    def __init__(self, root_dir, transform = lambda x, y: (x, y)):
        self.root_dir = root_dir   
        self.img_files = glob.glob(os.path.join(root_dir, '3band/*'))
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def expand(self, N):
        """
        Pad this dataset to have N elements
        """
        delta = N - len(self.img_files)
        to_add = []
        while delta > len(self.img_files):
            to_add.extend(self.img_files)
            delta -= len(self.img_files)
        to_add.extend(self.img_files[:delta])
        self.img_files.extend(to_add)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]

        vector_file = self.get_vector_file(img_file)
        vdata = json.load(open(vector_file, 'r'))

        if len(vdata['features']) == 0:
            return self[random.randint(0, len(self) - 1)]

        img_data = cv2.imread(img_file)
        boxes = []
        for feature in vdata['features']:
            geom = shape(feature['geometry'])
            bounds = geom.bounds
            minx, miny, maxx, maxy = self.get_bounds(img_file, bounds)
            boxes.append([minx, miny, maxx, maxy, 0])

        targets = np.array(boxes)

        mask = ((targets[:, 2] - targets[:, 0]) > 3) & ((targets[:, 3] - targets[:, 1]) > 3)
        targets = targets[mask, :]

        if len(targets) == 0 or img_data.shape[0] <= 300 or img_data.shape[1] <= 300:
            return self[random.randint(0, len(self) - 1)]

        ridx = random.randint(0, len(targets) - 1)

        cx, cy = round(np.mean(targets[ridx, (0, 2)])), round(np.mean(targets[ridx, (1, 3)]))
        minx, miny, maxx, maxy = cx - 150, cy - 150, cx + 150, cy + 150

        if minx < 0:
            dx = -minx
            minx, maxx = minx + dx, maxx + dx
        if maxx > img_data.shape[1]:
            dx = maxx - img_data.shape[1]
            minx, maxx = minx - dx, maxx - dx
        if miny < 0:
            dy = -miny
            miny, maxy = miny + dy, maxy + dy
        if maxy > img_data.shape[0]:
            dy = maxy - img_data.shape[0]
            miny, maxy = miny - dy, maxy - dy

        targets[:, (0, 2)] -= minx
        targets[:, (1, 3)] -= miny

        targets = np.clip(targets, a_min = 0, a_max = 300)

        mask = ((targets[:, 2] - targets[:, 0]) > 3) & ((targets[:, 3] - targets[:, 1]) > 3)

        targets = targets[mask, :]

        sample = img_data[int(miny):int(maxy), int(minx):int(maxx), :]

        if sample.shape[0] != 300 and sample.shape[1] != 300:
            print('Wrong dimensions!')
            pdb.set_trace()

        input_, targets, labels = self.transform(sample, targets[:, :4], targets[:, -1])

        if len(targets) == 0:
            print('No boxes!')
            pdb.set_trace()

        return (
            torch.from_numpy(input_.transpose((2,0,1))[(2,1,0),:,:]),
            np.hstack((targets, np.expand_dims(labels, axis=1)))
        )

class ProjDataset(SatelliteDataset):
    def get_bounds(self, filename, bounds):
        ds = gdal.Open(filename)
        minx, maxy = proj_to_raster(ds, *bounds[:2])
        maxx, miny = proj_to_raster(ds, *bounds[2:])
        return minx, miny, maxx, maxy

    def get_vector_file(self, img_file):
        img_id = re.search('img(\d+).tif', img_file).group(1)
        return os.path.join(self.root_dir, 'vectordata/geojson/Geo_AOI_1_RIO_img%s.geojson' % img_id)

class RasterDataset(SatelliteDataset):
    def get_bounds(self, filename, bounds):
        return bounds

    def get_vector_file(self, img_file):
        return img_file.replace('3band', 'vectordata').replace('.jpg', '.geojson')

class Dataset(data.Dataset):
    def __init__(self, transform = lambda x, y: (x, y)):
        self.spacenet = ProjDataset('processedBuildingLabels', transform)
        self.aigh_labeled = RasterDataset('training_data', transform)

        N = max(len(self.spacenet), len(self.aigh_labeled))
        self.spacenet.expand(N)
        self.aigh_labeled.expand(N)
        self.name = 'Spacenet + AIGH Labled Images'

    def __len__(self):
        return len(self.spacenet) + len(self.aigh_labeled)

    def __getitem__(self, idx):
        if idx < len(self.spacenet):
            return self.spacenet[idx]
        else:
            return self.aigh_labeled[idx - len(self.spacenet)]

class InferenceGenerator:
    def __init__(self, conn, country, transform = lambda x: x):
        self.conn = conn
        self.ts = datetime.now().isoformat()
        self.s3 = boto3.client('s3')
        self.country = country
        self.transform = transform

    def __iter__(self):
        cur = self.conn.cursor()
        write_cur = self.conn.cursor()
        #cur.execute("SELECT filename FROM buildings.images WHERE project=%s", (self.country,))

        cur.execute("""
            SELECT filename FROM buildings.images
            JOIN osm_buildings ON ST_Contains(images.shifted, osm_buildings.geom)
            WHERE images.project=%s
            GROUP BY filename HAVING COUNT(*) > 3
        """, (self.country,))

        for filename, in cur:
            params = {'Bucket' : 'dg-images', 'Key' : filename}

            url = self.s3.generate_presigned_url(ClientMethod='get_object', Params=params)
            img = io.imread(url)[:,:,(2,1,0)] # RGB -> BGR (transform, then convert back)

            for i in range(0, img.shape[0], 300):
                for j in range(0, img.shape[1], 300):
                    x, y = i, j
                    if i+300 > img.shape[0]:
                        x = img.shape[0] - 300
                    if j + 300 > img.shape[1]:
                        y = img.shape[1] - 300
                    orig = img[x:x+300, y:y+300, :]
                    yield (
                        torch.from_numpy(self.transform(orig.copy().astype(float)).transpose((2,0,1))[(2,1,0),:,:]).float(),
                        orig,
                        (x, y, filename)
                    )









