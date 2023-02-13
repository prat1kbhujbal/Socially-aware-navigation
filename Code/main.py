from dataset import Dataset
from model import Model


def main():
    dataset = Dataset()
    dataset.get_dataset()
    dataset.filter_images()

    model = Model()
    x, y = model.load_images()
    # model.show_image(x[0], y[0])
    model.output(x, y)


if __name__ == "__main__":
    main()
