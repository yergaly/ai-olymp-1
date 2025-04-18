import numpy as np
from io import StringIO
from scipy.ndimage import sobel
from scipy.stats import entropy
from scipy.ndimage import convolve, gaussian_filter, percentile_filter, median_filter

from sklearn.decomposition import PCA


def extract_features(a):
    def cg(i):
        gx = convolve(i, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode='reflect')
        gy = convolve(i, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), mode='reflect')
        return gx, gy

    def ch(i, cs=(4, 4), bs=(2, 3), bn=5):
        gx, gy = cg(i)
        m = np.sqrt(gx ** 2 + gy ** 2)
        o = np.arctan2(gy, gx) * (180 / np.pi) % 180
        ncx = i.shape[1] // cs[1]
        ncy = i.shape[0] // cs[0]
        h = np.zeros((ncy, ncx, bn))
        bw = 180 // bn

        for j in range(bn):
            mk = (o >= j * bw) & (o < (j + 1) * bw)
            h[:, :, j] = convolve(m * mk, np.ones(cs), mode='constant')[::cs[0], ::cs[1]]
        nbx = (ncx - bs[1]) + 1
        nby = (ncy - bs[0]) + 1
        nb = np.zeros((nby, nbx, bs[0], bs[1], bn))

        for y in range(nby):
            for x in range(nbx):
                b = h[y:y + bs[0], x:x + bs[1], :]
                nb[y, x] = b / np.sqrt(np.sum(b ** 2) + 1e-5)

        return nb.ravel()

    fl = []
    for i in a:
        i = gaussian_filter(i, sigma=1)
        i = median_filter(i, size=3)
        i = percentile_filter(i, percentile=65, size=3)

        i = (i - i.min()) / (i.max() - i.min() + 1e-5)
        fl.append(ch(i))


    pca = PCA(n_components=20, svd_solver='auto', random_state=1)
    X = pca.fit_transform(fl)
    return X


def extract_features0(arr):
    a = []
    for i in arr:
        mean = i.mean()
        std = i.std()
        median = np.median(i)
        iqr = np.percentile(i, 80) - np.percentile(i, 20)
        contrast = i.max() - i.min()
        sx, sy = sobel(i, axis=0), sobel(i, axis=1)
        mag = np.hypot(sx, sy)
        edge_mean = mag.mean()
        edge_std = mag.std()
        edge_density = (mag > 30).mean()
        hist = np.histogram(i, bins=32, range=(0, 255))[0] + 1e-6
        ent = entropy(hist)
        fg_ratio = np.mean(i > 50)
        ink = np.sum(i > 100) / 784
        total = np.sum(i) + 1e-6
        cx = np.sum(np.arange(28) * i.sum(axis=0)) / total
        cy = np.sum(np.arange(28) * i.sum(axis=1)) / total
        cent_dist = np.hypot(cx - 14, cy - 14)
        co = np.argwhere(i > 50)
        if co.size > 0:
            y_min, x_min = co.min(axis=0)
            y_max, x_max = co.max(axis=0)
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            aspect = w / (h + 1e-6)
            area_ratio = (h * w) / 784
        else:
            aspect = 0
            area_ratio = 0
        z = []
        for y in [0, 1, 2]:
            bb = i[y * 28 // 3:(y + 1) * 28 // 3, 10:18]
            z.append(bb.mean())
        for x in [0, 1, 2]:
            bb = i[10:18, x * 28 // 3:(x + 1) * 28 // 3]
            z.append(bb.mean())
        b = [mean, std, median, iqr, contrast,
             edge_mean, edge_std, edge_density, ent,
             fg_ratio, ink, cent_dist, aspect, area_ratio] + z[:6]
        a.append(b)
    return a


def get_total_input(text_lines):
    records = text_lines.strip().split("\n\n")
    N = int(records[0])
    records = records[1:]

    matrices = []
    for i in range(N):
        matrix = np.loadtxt(StringIO(records[i])).astype(int)
        matrices.append(matrix)
    return matrices


def float_list_to_str(float_list):
    string_list = [f"{num:.6f}" for num in float_list]
    return ' '.join(string_list)


def generate_total_output(matrices):
    features = extract_features(matrices)

    features_string_list = list(map(float_list_to_str, features))

    return '\n'.join(features_string_list)


input_txt = open("input.txt", "r").read()

matrices = get_total_input(input_txt)

feature_string = generate_total_output(matrices)

with open("output.txt", "w") as f:
    f.write(feature_string)
