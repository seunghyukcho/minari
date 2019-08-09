from minari.dataset import SolarDataSet


class TwoLevelDataSet(SolarDataSet):
    def set_data(self, data):
        self.x1_data, self.x2_data = data[:, [1, 4, 5, 7, 8, 9]], data[:, [2, 3, 6]]
        self.y_data = data[:, [0]]

    def __getitem__(self, item):
        return self.x1_data[item], self.x2_data[item], self.y_data[item]
