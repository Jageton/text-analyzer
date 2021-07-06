from util.visualization import dataframe_2d_vis


def show_all_projection(data_frame, predict, target_col_name):
    target_col_idx = 0
    for idx, name in enumerate(data_frame.columns):
        if str(name) == str(target_col_name):
            target_col_idx = idx

    for idx, name in enumerate(data_frame.columns):
        if str(name) != str(target_col_name):
            dataframe_2d_vis(data_frame, idx, target_col_idx, predict)

#          метод для списка индексов