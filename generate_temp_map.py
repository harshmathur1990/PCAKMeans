
def prepare_populate_temp_map(
    atmod, labels, tau_index=-2
):
    temp = atmod['temp'][0, 0, :, tau_index]

    def from_func(t, i, j):
        return temp[labels[int(t), int(i), int(j)]]

    return from_func
