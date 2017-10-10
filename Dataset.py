
from torch.utils import data
import glob, pdb, os, re, json, gdal, random, cv2, numpy as np, torch
from shapely.geometry import shape

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

class Dataset(data.Dataset):
    def __init__(self, root_dir, transform = lambda x, y: (x, y)):
        self.root_dir = root_dir   
        self.img_files = glob.glob(os.path.join(root_dir, '3band/*.tif'))
        self.transform = transform
        self.name = 'Spacenet'

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_id = re.search('img(\d+).tif', img_file).group(1)
        vector_file = os.path.join(self.root_dir, 'vectordata/geojson/Geo_AOI_1_RIO_img%s.geojson' % img_id)
        vdata = json.load(open(vector_file, 'r'))

        if len(vdata['features']) == 0:
            return self[random.randint(0, len(self) - 1)]

        ds = gdal.Open(img_file)

        img_data = cv2.imread(img_file)

        boxes = []

        for feature in vdata['features']:
            geom = shape(feature['geometry'])
            bounds = geom.bounds
            minx, maxy = proj_to_raster(ds, *bounds[:2])
            maxx, miny = proj_to_raster(ds, *bounds[2:])
            boxes.append([minx, miny, maxx, maxy, 0])

        targets = np.array(boxes)

        mask = ((targets[:, 2] - targets[:, 0]) > 3) & ((targets[:, 3] - targets[:, 1]) > 3)
        targets = targets[mask, :]

        if len(targets) == 0:
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

        sample = img_data[int(minx):int(maxx), int(miny):int(maxy), :]

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
        

































