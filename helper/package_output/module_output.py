# ==================================================================================================================== #
# Для расчетов
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# Для таблиц
import csv

# ==================================================================================================================== #


def print_result(f, result, sep=''):
    print(len(result), ":")
    for i in range(len(result)):
        print(i, sep, result[i], sep, f(result[i]))

# -------------------------------------------------------------------------------------------------------------------- #


def save_result_text(f, result, filename, sep=' ', sp='%g'):
    with open(filename, 'w') as file:
        for data in result:
            for x in data:
                file.write((sp + "%s") % (x, sep))
            file.write((sp + "\n") % f(data))


# -------------------------------------------------------------------------------------------------------------------- #


def save_result_table(f, result, filename, sp='%g', fields=None, generate_fields=False):
    if fields is None:
        fields = []
    with open(filename, 'w') as csvfile:
        # создание объекта writer csv
        csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONE)

        if generate_fields:
            if len(result[0]) == 1:
                fields = ['X']
            elif len(result[0]) == 2:
                fields = ['X', 'Y']
            else:
                fields = [f"X[{i}]" for i in range(len(result[0]))]
            fields.append('F')

        # запись шапки
        if len(fields) > 0:
            csvwriter.writerow(fields)

            # запись данных
        data = np.insert(result, len(result[0]), [f(x) for x in result], axis=1)
        formatted_data = [[sp % x for x in row] for row in data]
        csvwriter.writerows(formatted_data)

# -------------------------------------------------------------------------------------------------------------------- #


def save_result(f, list_result, list_label, filepath='', sp='%g', fields=None, generate_fields=True):
    for i in range(len(list_result)):
        save_result_text(f, list_result[i], filepath + list_label[i] + '.txt', sp=sp)
        save_result_table(f, list_result[i], filepath + list_label[i] + '.csv', sp, fields, generate_fields)

# ==================================================================================================================== #
