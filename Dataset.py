
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch, cv2, boto3
from torch.utils import data
import glob, pdb, os, re, json, random, numpy as np, torch
from shapely.geometry import shape
from datetime import datetime
from skimage import io

SIZE = 300

def RandomSampler(conn, country, transform):
    ts = datetime.now().isoformat()
    s3 = boto3.client('s3')
    read_cur = conn.cursor()
    write_cur = conn.cursor()

    read_cur.execute("""
        SELECT filename, ST_AsGeoJSON(shifted)::json FROM buildings.images
        WHERE project=%s AND (done IS NULL OR done=false)
        ORDER BY random()
    """, (country,))

    for filename, geom in read_cur:
        params = {'Bucket' : 'dg-images', 'Key' : filename}
        url = s3.generate_presigned_url(ClientMethod='get_object', Params=params)
        # Convert from RGB -> BGR and also strip off the bottom logo

        img = io.imread(url)[:-25,:,(2,1,0)]

        for i in range(0, img.shape[0], SIZE):
            for j in range(0, img.shape[1], SIZE):
                x, y = i, j
                if i+SIZE > img.shape[0]:
                    x = img.shape[0] - SIZE
                if j + SIZE > img.shape[1]:
                    y = img.shape[1] - SIZE
                orig = img[x:x+SIZE, y:y+SIZE, :]
                transformed, _, _ = transform(orig.copy().astype(float))
                yield (
                    torch.from_numpy(transformed.transpose((2,0,1))[(2,1,0),:,:]).float(),
                    orig,
                    (x, y, filename, img, shape(geom))
                )

commit_thunk = lambda: True

def InferenceGenerator2(conn, country, TS, min_ts = None, area_to_cover = None, transform=lambda x: x, num_workers=1, data_dir='./', cache=False):
    ts = datetime.now().isoformat()

    condition = ''
    if area_to_cover:
        condition = " AND ST_Contains(ST_GeomFromText('%s', 4326), shifted)" % area_to_cover.wkt

    # if min_ts:
    condition += " AND last_tested IS NULL" #>= '%s'" % min_ts.isoformat()

    cur = conn.cursor()

    cur.execute("""
        SELECT filename FROM buildings.images
        WHERE project=%%s %s
    """ % condition, (country,))

    files = cur.fetchall()
    


def InferenceGenerator(conn, country, TS, min_ts = None, area_to_cover = None, transform=lambda x: x, data_dir='./', cache=False):
    global commit_thunk
    ts = datetime.now().isoformat()
    s3 = boto3.client('s3')

    condition = ''
    if area_to_cover:
        condition = " AND ST_Contains(ST_GeomFromText('%s', 4326), shifted)" % area_to_cover.wkt

    # if min_ts:
    condition += " AND last_tested IS NULL" #>= '%s'" % min_ts.isoformat()

    def commit():
        global commit_thunk
        conn.commit()
        commit_thunk = lambda: True

    write_cur = conn.cursor()
    cur = conn.cursor()

    cur.execute("""
        SELECT filename FROM buildings.images
        WHERE project=%%s %s
    """ % condition, (country,))

    for filename, in cur:
        if os.path.exists(os.path.join(data_dir, filename)):
            img = cv2.imread(os.path.join(data_dir, filename))[:-25, :, :]
        else:
            while True:
                try:
                    params = {'Bucket' : 'dg-images', 'Key' : filename}
                    url = s3.generate_presigned_url(ClientMethod='get_object', Params=params)
                    # Convert from RGB -> BGR
                    img = io.imread(url)[:,:,(2,1,0)]
                    break
                except Exception as e:
                    print(e)

            if cache:
                if not os.path.exists(os.path.dirname(os.path.join(data_dir, filename))):
                    os.makedirs(os.path.dirname(os.path.join(data_dir, filename)))
                cv2.imwrite(os.path.join(data_dir, filename), img)

            # Strip off the bottom logo
            img = img[:-25, :, :]

        for i in range(0, img.shape[0], SIZE):
            for j in range(0, img.shape[1], SIZE):
                x, y = i, j
                if i+SIZE > img.shape[0]:
                    x = img.shape[0] - SIZE
                if j + SIZE > img.shape[1]:
                    y = img.shape[1] - SIZE
                orig = img[x:x+SIZE, y:y+SIZE, :]
                transformed, _, _ = transform(orig.copy())
                yield (
                    torch.from_numpy(transformed.transpose((2,0,1))[(2,1,0),:,:]).float(),
                    orig,
                    (x, y, filename),
                    commit_thunk
                )
        write_cur.execute("""
            UPDATE buildings.images SET last_tested=%s WHERE project=%s AND filename=%s
        """, (TS, country, filename))
        commit_thunk = commit


class Dataset(data.Dataset):
    def __init__(self, root_dir, samples, transform = lambda a1,a2,a3: (a1,a2,a3)):
        self.root_dir = root_dir   
        self.transform = transform
        self.samples = samples

    def even(self):
        projs = {}
        for sample in self.samples:
            if len(sample['rects']) > 0:
                proj = re.search('(.*[^\d])(\d+)\.', os.path.basename(sample['image_path'])).group(1)
                if proj in projs:
                    projs[proj].append(sample)
                else:
                    projs[proj] = [sample]

        samples = []

        max_size = max([len(projs[k]) for k in projs.keys()])
        for proj in projs.keys():
            arr = projs[proj]
            count = 0
            while count + len(arr) <= max_size and count / 10 < len(arr):
                samples.extend(arr)
                count += len(arr)
            diff = (max_size - len(arr)) % len(arr)
            print('Added %d samples from %s' % (diff + count, proj))
            random.shuffle(arr)
            samples.extend(arr[:diff])
        self.samples = samples
        return self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filename = sample['image_path']

        if len(sample['rects']) == 0:
            return self[random.randint(0, len(self) - 1)]

        img_data = cv2.imread(os.path.join(self.root_dir, sample['image_path']))
        if img_data is None:
            pdb.set_trace()
        boxes = []
        for f in sample['rects']:
            boxes.append([f['x1'], f['y1'], f['x2'], f['y2'], 0])

        targets = np.array(boxes).astype(float)

        mask = ((targets[:, 2] - targets[:, 0]) > 3) & ((targets[:, 3] - targets[:, 1]) > 3)
        targets = targets[mask, :]

        if img_data.shape[0] <= SIZE or img_data.shape[1] <= SIZE:
            return self[random.randint(0, len(self) - 1)]

        if len(targets) > 0:
            ridx = random.randint(0, len(targets) - 1)

            cx, cy = round(np.mean(targets[ridx, (0, 2)])), round(np.mean(targets[ridx, (1, 3)]))
            minx, miny, maxx, maxy = cx - (SIZE/2), cy - (SIZE/2), cx + (SIZE/2), cy + (SIZE/2)

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

            targets = np.clip(targets, a_min = 0, a_max = SIZE)

            mask = ((targets[:, 2] - targets[:, 0]) > 3) & ((targets[:, 3] - targets[:, 1]) > 3)

            targets = targets[mask, :]
        else:
            minx, miny = np.random.randint(img_data.shape[0] - SIZE), np.random.randint(img_data.shape[1] - SIZE)
            maxx, maxy = minx + SIZE, miny + SIZE
            targets = np.zeros((0, 5))

        sample = img_data[int(miny):int(maxy), int(minx):int(maxx), :]

        if sample.shape[0] != SIZE and sample.shape[1] != SIZE:
            print('Wrong dimensions!')
            pdb.set_trace()

        input_, targets, labels = self.transform(sample, targets[:, :4], targets[:, -1])
        
        return (
            torch.from_numpy(input_.transpose((2,0,1))[(2,1,0),:,:]).float(),
            np.hstack((targets, np.expand_dims(labels, axis=1)))
        )