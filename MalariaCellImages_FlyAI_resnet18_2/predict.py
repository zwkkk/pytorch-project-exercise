from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)

p = model.predict(image_path="images/C39P4thinF_original_IMG_20150622_105253_cell_111.png")

print(p)