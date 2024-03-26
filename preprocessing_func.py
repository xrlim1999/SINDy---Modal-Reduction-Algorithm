import csv
import numpy as np

def import_data(filename, rows, cols):
    """
    Import data from specified csv files
    """
    f = open(filename)
    data_csv = csv.reader(f) # read CSV file

    # define and populate training data matrix
    rows_csv = rows
    columns_csv = cols

    data = np.zeros((rows_csv, columns_csv))
    row_num = 0
    for row in data_csv:
        col_num = 0
        for now in row:
            data[row_num, col_num] = now
            col_num += 1
        row_num += 1

    f.close()

    return np.transpose(data)  # re-sahpe into vertical skinny matrix


def delete(x, colnums=None):
    """
    Delete specified data columns
    """
    xtrunc = x.copy()
    count = 0
    if colnums is not None:
        for i in range(len(colnums)):
            xtrunc = np.delete(xtrunc, colnums[i]-count, 1)
            count += 1

    return xtrunc


def xdot_generate(x, t):

    x_dot = [0] * (len(x))  # initialise x_dot matrix

    for i in range(len(x)-1):
        dt = t[i+1]-t[i]
        dx = x[i+1] - x[i-1]
        x_dot[i] = dx / (2*dt)

    # x_dot[0] = (x[1]-x[0]) / (2*(t[1]-t[0]))
    x_dot[-1] = (x[0]-x[-2]) / (2*(t[-1]-t[-2]))


    # x_dot = np.diff(x)/(t[1]-t[0])
    # x_dot = x_dot + [x[0]]

    return x_dot
