
from mnist_objective.Lighting import DataModule

if __name__ == '__main__':
    dm = DataModule(transform=None)
    dm.setup(stage='test')
    i = 677
    find = 8

    val = -1
    found = 0
    while found != 5:
        sample = dm.test[i]
        im, val = sample
        if find == val:
            found += 1
            print(f'{i}: {val}')
            im.show()
        i += 1

