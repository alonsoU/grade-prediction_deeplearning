def all_encodings():
    import os
    import pkgutil
    import encodings
    modnames = set([modname for importer, modname, ispkg in pkgutil.walk_packages(
        path=[os.path.dirname(encodings.__file__)], prefix='')])
    aliases = set(encodings.aliases.aliases.values())
    return modnames.union(aliases)

def spectre(paths):
#     m_area, m_height, m_width
    min = [(1e10,1e10), 1e10, 1e10]
    max = [(0,0), 0, 0]
    from PIL import Image
    for img in paths:
        image = Image.open(img)
        size = image.size
        height = size[0]
        width = size[1]
        area = height*width
        if area > max[0][0]*max[0][1]:
            max[0] = size
        if area < min[0][0]*min[0][1]:
            min[0] = size
        if height > max[1]:
            max[1] = height
        if height < min[1]:
            min[1] = height
        if width > max[2]:
            max[2] = width
        if width < min[2]:
            min[2] = width
    return min, max
"""
text = b'\0xd1'
for enc in all_encodings():
    try:
        msg = text.decode(enc)
    except Exception:
        continue
    if msg == 'ñ':
        print('Decoding {t} with {enc} is {m}'.format(t=text, enc=enc, m=msg))
"""
"""
# otro metodo no python de masking
column_mask = (dtypes_ == 'float') | (dtypes_ == 'int')
df.loc[:,~column_mask] = df.loc[:,~column_mask].applymap(lambda x: x.lower().strip())
"""
"""
# buen metodo para generar masks
mask = df.isnull().any(axis=1)
df = df.loc[~mask,:]
"""
def benchmark(dataset, num_epochs=2):
    import time
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)
