import matplotlib.pyplot as plt


# ==================================================================================================================== #

def print_2d_data(file_info_2d_data, results, lables, show_print=False, title='', filename='', filename_extension='.png'
                  , dpi=1024):
    if show_print:
        print("Коэффициенты уравнения прямой:")
        print(f"y = {file_info_2d_data.real_weight:.3f} * x + {file_info_2d_data.real_bias:.3f}")

    # ======== style-parameters
    plt.style.use('default')
    plt.figure(figsize=(8, 8))
    # =========================

    plt.scatter(file_info_2d_data.X, file_info_2d_data.y, color='gray', alpha=0.5, s=20.8)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.plot(file_info_2d_data.X, file_info_2d_data.real_weight * file_info_2d_data.X + file_info_2d_data.real_bias,
             label='Real', color='lime', linewidth=1.7)
    for i in range(len(results)):
        for j in range(len(file_info_2d_data.labels_loss)):

            full_label = lables[i] + ' ' + file_info_2d_data.labels_loss[j] + ' ' + str(len(results[i][j]))

            plt.plot(file_info_2d_data.X, results[i][j][-1][0] * file_info_2d_data.X + results[i][j][-1][1],
                     label=full_label,
                     linewidth=1.5)
            if show_print:
                print("Вычисленные коэффициенты уравнения прямой " + full_label + ":")
                print(f"y = {results[i][j][-1][0]:.3f} * x + {results[i][j][-1][1]:.3f}")

    if title != '':
        plt.title(title)

    plt.legend()

    if filename != '':
        plt.savefig(filename + filename_extension, dpi=dpi, bbox_inches=0, transparent=True)

    plt.show()

# ==================================================================================================================== #


def print_loss_history(results, labels):
    for i in range(len(results)):
        plt.plot(results[i], label=labels[i])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

# ==================================================================================================================== #
