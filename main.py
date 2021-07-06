import urllib.request
from enum import Enum
from os import getcwd
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter.ttk import Notebook

import sys

import pandas

from buffer import Buffer
from classification.decision_tree.decision_tree import DecisionTreeCriterion, DecisionTreeSplitter
from classification.knn.k_nearest_neighbors import KNNAlgorithmType, KNNWeightType
from classification.random_forests.random_forests import RandomForestCriterion
from clustering.agglomerative_clustering.agglomerative_clustering import AffinityType, LinkageType
from clustering.dbscan.dbscan import DBSCANMetricType, DBSCANAlgorithmType
from clustering.k_means.k_means import KMeansInitType, KMeansAlgorithmType
from clustering.spectral_clustering.spectral_clustering import EigenSolver
from pio.input import Input
from pio.output import Output
from runner import *
from util.visualization import data_3d_visualization

listbox = None


def main():
    window = Tk()
    window.title("Text analyzer")
    window.geometry('700x740')
    tab_control = Notebook(window, height=440, width=700)

    tab1 = ttk.Frame(tab_control)
    tab2 = ttk.Frame(tab_control)
    tab3 = ttk.Frame(tab_control)
    tab4 = ttk.Frame(tab_control)
    tab5 = ttk.Frame(tab_control)
    tab6 = ttk.Frame(tab_control)
    tab7 = ttk.Frame(tab_control)
    tab8 = ttk.Frame(tab_control)
    tab9 = ttk.Frame(tab_control)
    tab10 = ttk.Frame(tab_control)

    log_tab = ttk.Frame(tab_control)

    tab_control.add(tab1, text='KNN')
    tab_control.add(tab2, text='Naive Bayes')
    tab_control.add(tab3, text='Decision Tree')
    tab_control.add(tab4, text='Random forest')
    tab_control.add(tab5, text='DB SCAN')
    tab_control.add(tab6, text='Birch')
    tab_control.add(tab7, text='Agglomerative')
    tab_control.add(tab8, text='Spectral clustering')
    tab_control.add(tab9, text='K Means')
    tab_control.add(tab10, text='Mean Shift')

    tab_control.add(log_tab, text='Log')
    tab_control.place(x=0, y=300)

    knn_tab(tab1)
    naive_bayes_tab(tab2)
    decision_tree_tab(tab3)
    random_forest_tab(tab4)

    db_scan_tab(tab5)
    birch_tab(tab6)
    agglomerative_tab(tab7)
    spectral_clustering_tab(tab8)
    k_means_tab(tab9)
    mean_shift_tab(tab10)
    fill_log_tab(log_tab)

    label = Label(window, text="Ресурсы")
    label.place(x=10, y=10)

    global listbox
    listbox = Listbox(window, width=40, height=10)
    listbox.place(x=10, y=40)

    add_file_button = Button(window, text="Добавить файл",
                             command=lambda: add_file_button_clicked(listbox))
    add_file_button.place(x=370, y=40)

    delete_file_button = Button(
        window, text="Удалить", command=lambda: delete_file_button_clicked(listbox))
    delete_file_button.place(x=370, y=140)

    def _show_url_input():
        new_window = Toplevel(window)
        Label(new_window, text="URL").pack()
        url = Entry(new_window, width=30)
        url.pack()

        def _exit():
            if url.get() != "":
                listbox.insert(END, url.get())

            new_window.destroy()
            new_window.update()

        def _check():
            try:
                conn = urllib.request.Request(url.get())
                conn.get_method = lambda: 'HEAD'
                response = urllib.request.urlopen(conn)
                if response.status == 200:
                    messagebox.showinfo(
                        title='Проверка подключения', message='URL существует')
                else:
                    messagebox.showerror(
                        title='Проверка подключения', message='URL не существует')
            except ValueError:
                messagebox.showerror(
                    title='Проверка подключения', message='URL не существует')
            except Exception:
                messagebox.showerror(
                    title='Проверка подключения', message='URL не существует')

        Button(new_window, text="Проверить", command=_check).pack()
        Button(new_window, text="Добавить", command=_exit).pack()

    Button(window, text="Добавить URL", command=_show_url_input).place(x=370, y=90)

    window.mainloop()


def catch_std_out(log):
    sys.stdout = Buffer(log)


def add_file_button_clicked(listbox: Listbox):
    filename = filedialog.askopenfilename(initialdir=getcwd())
    listbox.insert(END, filename)


def delete_file_button_clicked(listbox: Listbox):
    index = listbox.curselection()
    if index != ():
        listbox.delete(index)


def birch_tab(frame):
    threshold = add_input(10, 30, frame, "Threshold", 0.5)
    branching_factor = add_input(10, 80, frame, "Branching factor", 50)
    n_clusters = add_input(10, 130, frame, "N clusters", 3)
    nsamples = add_input(10, 290, frame, "Nsamples", 10000)

    def _start_with_nsamples():
        run_birch(int(nsamples.get()), float(threshold.get()), int(branching_factor.get()), int(n_clusters.get()))

    def _start_with_file():
        def _start():
            alg = Birch(
                threshold=float(threshold.get()),
                branching_factor=int(branching_factor.get()),
                n_clusters=int(n_clusters.get())
            )
            return lambda df: (get_attributes(alg), alg.run(df))

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _start)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def db_scan_tab(frame):
    eps = add_input(10, 30, frame, "Eps", 0.5)
    metric = add_input(10, 80, frame, "Metric", DBSCANMetricType)
    algorithm = add_input(10, 130, frame, "Algorithm", DBSCANAlgorithmType)
    leaf_size = add_input(150, 30, frame, "Leaf size", 30)
    p = add_input(150, 80, frame, "Minkowski metric power", "2 or None")
    nsamples = add_input(10, 290, frame, "Nsamples", 10000)

    def _start_with_nsamples():
        power = p.get() != "2" if None else 2
        run_db_scan(nsamples=int(nsamples.get()), eps=float(eps.get()), metric=metric.get(),
                    algorithm=algorithm.get(),
                    leaf_size=int(leaf_size.get()),
                    p=power)

    def _start_with_file():
        def _start():
            alg = DBSCAN(
                eps=float(eps.get()), metric=metric.get(), algorithm=algorithm.get(),
                leaf_size=int(leaf_size.get()), p=power
            )
            return lambda df: (get_attributes(alg), alg.run(df))

        power = p.get() != "2" if None else 2
        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _start)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def agglomerative_tab(frame):
    linkage = add_input(10, 30, frame, "linkage", LinkageType)
    n_clusters = add_input(10, 80, frame, 'n_clusters', 2)
    affinity = add_input(10, 130, frame, "affinity", AffinityType)
    nsamples = add_input(10, 290, frame, "Nsamples", 10000)

    def _start_with_nsamples():
        run_agglomerative(nsamples=int(nsamples.get()),
                          affinity=affinity.get(),
                          linkage=linkage.get(),
                          n_clusters=int(n_clusters.get()))

    def _start_with_file():
        def _start():
            alg = AgglomerativeClustering(
                n_clusters=int(n_clusters.get()),
                linkage=linkage.get(),
                affinity=affinity.get() if linkage.get() != 'ward' else AffinityType.euclidean.value
            )
            return lambda df: (get_attributes(alg), alg.run(df))

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _start)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def spectral_clustering_tab(frame):
    n_clusters = add_input(10, 30, frame, 'n_clusters', 2)
    eigen_solver = add_input(10, 80, frame, "eigen_solver", EigenSolver)
    nsamples = add_input(10, 290, frame, "Nsamples", 10000)

    def _start_with_nsamples():
        run_spectral_clustering(int(nsamples.get()), int(n_clusters.get()), eigen_solver.get())

    def _start_with_file():
        def _start():
            alg = SpectralCluster(n_clusters=int(n_clusters.get()),
                                  eigen_solver=eigen_solver.get())
            return lambda df: (get_attributes(alg), alg.run(df))

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _start)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def k_means_tab(frame):
    n_clusters = add_input(10, 30, frame, 'n_clusters', 2)
    init = add_input(10, 80, frame, "init", KMeansInitType)
    algorithm = add_input(10, 140, frame, "algorithm", KMeansAlgorithmType)
    n_init = add_input(10, 180, frame, "n_init", 10)
    max_iter = add_input(10, 220, frame, "max_iter", 300)
    nsamples = add_input(10, 290, frame, "Nsamples", 10000)

    def _start_with_nsamples():
        run_kmeans(int(nsamples.get()), int(n_clusters.get()), init.get(),
                   algorithm.get(), int(n_init.get()), int(max_iter.get()))

    def _start_with_file():
        def _start():
            alg = KMeans(
                n_clusters=int(n_clusters.get()), init=init.get(), algorithm=algorithm.get(),
                n_init=int(n_init.get()), max_iter=int(max_iter.get())
            )
            return lambda df: (get_attributes(alg), alg.run(df))

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _start)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def knn_tab(frame):
    n_neighbors = add_input(10, 30, frame, 'n_neighbors', 5)
    algorithm = add_input(10, 70, frame, 'algorithm', KNNAlgorithmType)
    weight = add_input(10, 120, frame, 'weight', KNNWeightType)
    test_size = StringVar()

    Scale(
        frame, from_=0.1, to=0.9, length=90, showvalue=1, variable=test_size,
        digits=2, resolution=0.05, orient="horizontal", label='test_size'
    ).place(x=10, y=170)

    def _start_with_nsamples():
        run_knearest_neighbors(n_neighbors=int(n_neighbors.get()),
                               algorithm=algorithm.get(),
                               weight=weight.get(),
                               test_size=float(test_size.get()))

    def _start_with_file():
        def _s():
            return lambda df: _start(df)

        def _start(df):
            y = df['target'].values
            x = df.drop(columns=['target'])
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=float(test_size.get()), random_state=1)
            alg = KNN(
                n_neighbors=int(n_neighbors.get()),
                algorithm=algorithm.get(),
                weights=weight.get()
            )
            return get_attributes(alg), alg.run(train_x=x_train, train_y=y_train, test_x=x_test)

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _s, clusterization=False)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def naive_bayes_tab(frame):
    test_size = StringVar()
    Scale(
        frame, from_=0.1, to=0.9, length=90, showvalue=1, variable=test_size,
        digits=2, resolution=0.05, orient="horizontal", label='test_size'
    ).place(x=10, y=40)
    var_smoothing = add_input(10, 140, frame, "var_smoothing", "0.00000009")

    def _start_with_nsamples():
        run_naive_bayes(test_size=float(test_size.get()))

    def _start_with_file():
        def _s():
            return lambda df: _start(df)

        def _start(df):
            y = df['target'].values
            x = df.drop(columns=['target'])
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=float(test_size.get()), random_state=1)
            alg = NaiveBayes(
                var_smoothing=float(var_smoothing.get())
            )
            return get_attributes(alg), alg.run(x_train=x_train, y_train=y_train, x_test=x_test)

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _s, clusterization=False)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def decision_tree_tab(frame):
    criterion = add_input(10, 30, frame, 'Criterion', DecisionTreeCriterion)
    splitter = add_input(10, 80, frame, "Splitter", DecisionTreeSplitter)
    test_size = StringVar()
    Scale(
        frame, from_=0.1, to=0.9, length=90, showvalue=1, variable=test_size,
        digits=2, resolution=0.05, orient="horizontal", label='test_size'
    ).place(x=10, y=130)

    def _start_with_nsamples():
        run_decision_tree(splitter=splitter.get(), criterion=criterion.get(), test_size=float(test_size.get()))

    def _start_with_file():
        def _s():
            return lambda df: _start(df)

        def _start(df):
            y = df['target'].values
            x = df.drop(columns=['target'])
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=float(test_size.get()), random_state=1)
            alg = DecisionTree(
                criterion=criterion.get(), splitter=splitter.get()
            )
            return get_attributes(alg), alg.run(train_x=x_train, train_y=y_train, test_x=x_test)

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _s, clusterization=False)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def random_forest_tab(frame):
    n_estimators = add_input(10, 30, frame, "n_estimators", 100)
    criterion = add_input(10, 80, frame, "criterion", RandomForestCriterion)
    verbose = add_input(10, 130, frame, 'verbose', 0)
    random_state = add_input(10, 180, frame, "random state", 1)
    test_size = StringVar()

    Scale(
        frame, from_=0.1, to=0.9, length=90, showvalue=1, variable=test_size,
        digits=2, resolution=0.05, orient="horizontal", label='test_size'
    ).place(x=140, y=30)

    def _start_with_nsamples():
        run_random_forest(n_estimators=int(n_estimators.get()),
                          criterion=criterion.get(),
                          verbose=int(verbose.get()),
                          random_state=int(random_state.get()),
                          test_size=float(test_size.get()))

    def _start_with_file():
        def _s():
            return lambda df: _start(df)

        def _start(df):
            y = df['target'].values
            x = df.drop(columns=['target'])
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=float(test_size.get()), random_state=1)
            alg = RandomForest(
                n_estimators=int(n_estimators.get()),
                criterion=criterion.get(),
                verbose=int(verbose.get()),
                random_state=int(random_state.get())
            )
            return get_attributes(alg), alg.run(x_train=x_train, y_train=y_train, x_test=x_test)

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _s, clusterization=False)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def mean_shift_tab(frame):
    max_iter = add_input(10, 30, frame, "max_iter", 300)
    bin_seeding = BooleanVar()
    cluster_all = BooleanVar()
    Checkbutton(frame, text="bin_seeding", variable=bin_seeding).place(x=100, y=30)
    Checkbutton(frame, text="cluster_all", variable=cluster_all).place(x=100, y=50)

    nsamples = add_input(10, 290, frame, "Nsamples", 10000)

    def _start_with_nsamples():
        run_mean_shift(max_iter=int(max_iter.get()), bin_seeding=bin_seeding.get(),
                       cluster_all=cluster_all.get(), nsamples=int(nsamples.get()))

    def _start_with_file():
        def _start():
            alg = MeanShift(
                max_iter=int(max_iter.get()), bin_seeding=bin_seeding.get(), cluster_all=cluster_all.get()
            )
            return lambda df: (get_attributes(alg), alg.run(df))

        dataframe = read_active_file()
        if dataframe is not None:
            run_algorithm(frame, dataframe.copy(), _start)

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def run_algorithm(frame, dataframe, algorithm, clusterization=True):
    new_window = Toplevel(frame)
    new_window.columnconfigure(0, pad=10)
    new_window.columnconfigure(1, pad=10)
    new_window.columnconfigure(2, pad=10)
    new_window.rowconfigure(0, pad=10)
    new_window.rowconfigure(1, pad=10)
    new_window.rowconfigure(2, pad=20)

    Label(new_window, text="Первый столбец*").grid(column=0, row=0)
    combo_box1 = ttk.Combobox(new_window,
                              values=[i for i in dataframe.columns], width=10)
    combo_box1.grid(column=0, row=1)
    combo_box1.current(0)

    Label(new_window, text="Второй столбец*").grid(column=1, row=0)
    combo_box2 = ttk.Combobox(new_window,
                              values=[i for i in dataframe.columns], width=10)
    combo_box2.grid(column=1, row=1)
    combo_box2.current(1)

    Label(new_window, text="Третий столбец").grid(column=2, row=0)
    combo_box3 = ttk.Combobox(new_window,
                              values=['None'] + [i for i in dataframe.columns], width=10)
    combo_box3.insert(0, 'None')
    combo_box3.grid(column=2, row=1)
    combo_box3.current(0)

    def _start():
        df = dataframe.copy()
        for name in df.columns:
            if combo_box1.get() != name and combo_box2.get() != name and combo_box3.get() != name:
                del df[name]

        alg = algorithm()
        (params, predict) = alg(dataframe)
        if clusterization:
            if len(df.columns) == 2:
                data_2d_visualization(df, predict)
            else:
                data_3d_visualization(df, predict)

        is_yes = messagebox.askyesno(message="Сохранить результат в Excel?")
        if is_yes:
            save_to_excel(dataframe, predict, params, clusterization)

    Button(new_window, text='Запустить', command=_start).grid(column=1, row=2)


def save_to_excel(data_frame, predict, params, clusterization):
    file = filedialog.asksaveasfile(initialdir=getcwd())
    file_name = file.name
    file.close()
    if clusterization:
        save_clusterization_result_to_excel(file_name, data_frame, predict, params)
    else:
        save_classification_result_to_excel(file_name, data_frame, predict, params)


def add_input(x, y, frame, text, value):
    Label(frame, text=text).place(x=x, y=y)
    if isinstance(value, (str, int, float)):
        entry = Entry(frame, width=10)
        entry.insert(0, value)
        entry.place(x=x, y=y + 20)
        return entry
    elif issubclass(value, Enum):
        combo_box = ttk.Combobox(frame, values=[i.value for i in value], width=10)
        combo_box.current(0)
        combo_box.place(x=x, y=y + 20)
        return combo_box
    return None


def read_file():
    path = listbox.get(ACTIVE)
    if path == "":
        messagebox.showerror(message='Не был выбран файл или URL')
    elif path.startswith('http'):
        if path.endswith('csv'):
            return Input.internet_read_csv(path)
        return Input.internet_read_text_file(path)
    elif path.endswith('xls') or path.endswith('xlsx'):
        return pandas.read_excel(path)
    else:
        if path.endswith('csv'):
            return Input.local_read_csv(path)
        return Input.local_read_text_file(path)


def read_active_file():
    try:
        path = listbox.get(ACTIVE)
        if path != "":
            if path.endswith('xls') or path.endswith('xlsx'):
                excel = pandas.read_excel(path)
                if type(excel) is DataFrame:
                    return excel
                else:
                    return excel[0]
            return pandas.read_csv(path)
        else:
            messagebox.showerror(message='Не был выбран файл или URL')
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        messagebox.showerror(
            message="Произошла ошибочка. Посмотрите в лог.")
    return None


def save_clusterization_result_to_excel(path_to_save, data_frame, prediction, params):
    data_frame_rows = data_frame.values
    rows = [data_frame_rows[i] for i in range(len(data_frame_rows))]
    dictionary = {}
    for i in range(max(prediction) + 1):
        indexes = [j for j, x in enumerate(prediction) if x == i]
        points = {i + 1: rows[i] for i in indexes}
        dictionary['Кластер %d:' % (i + 1)] = points

    Output.write_to_xlsx_file(path_to_save, data_frame, dictionary, params)


def save_classification_result_to_excel(path_to_save, data_frame, prediction, params):
    iris_frame_rows = data_frame.values
    rows = [iris_frame_rows[i] for i in range(len(iris_frame_rows))]
    dictionary = {}
    for i in range(max(prediction) + 1):
        indexes = [j for j, x in enumerate(prediction) if x == i]
        points = {i + 1: rows[i] for i in indexes}
        dictionary['Класс %d:' % (i + 1)] = points

    Output.write_to_xlsx_file(path_to_save, data_frame, dictionary, params)


def get_attributes(alg):
    attributes = dict((k, v) for (k, v) in alg.__dict__.items() if not str(k).startswith('__') and v is not None)
    attributes['alg_name'] = alg.__class__.__name__
    return attributes


def fill_log_tab(log_tab):
    log = Text(log_tab, width=87, height=26)
    log.place(x=0, y=0)
    catch_std_out(log)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
