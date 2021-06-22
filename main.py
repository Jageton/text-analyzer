# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import *
from tkinter import ttk, filedialog

from os import getcwd
from tkinter.ttk import Notebook

from buffer import Buffer
from clustering.dbscan.dbscan import DBSCANMetricType
from runner import *

alg = 0

params = {}


def main():
    global alg

    window = Tk()
    window.title("Text analyzer")
    window.geometry('700x740')
    tabControl = Notebook(window, height=440, width=700)

    tab1 = ttk.Frame(tabControl)
    tab2 = ttk.Frame(tabControl)

    tabControl.add(tab1, text='DB SCAN')
    tabControl.add(tab2, text='Birch')
    tabControl.place(x=0, y=300)

    dbScanTab(tab1)

    ttk.Label(tab2,
              text="Lets dive into the\
              world of computers").grid(column=0,
                                        row=0,
                                        padx=30,
                                        pady=30)

    label = Label(window, text="Ресурсы")
    label.place(x=10, y=10)

    listbox = Listbox(window, width=40, height=10)
    listbox.place(x=10, y=40)

    add_file_button = Button(window, text="Добавить файл", command=lambda: add_file_button_clicked(listbox))
    add_file_button.place(x=370, y=40)

    delete_file_button = Button(window, text="Удалить", command=lambda: delete_file_button_clicked(listbox))
    delete_file_button.place(x=370, y=90)

    # Label(window, text="Count of points").place(x=20, y=230)
    #
    # nsamples_entry = Entry(window, width=10)
    # nsamples_entry.insert(0, "1000")
    # nsamples_entry.place(x=20, y=250)
    #
    # Label(window, text="Cluster std").place(x=20, y=275)
    #
    # cluster_std_entry = Entry(window, width=10)
    # cluster_std_entry.insert(0, 1)
    # cluster_std_entry.place(x=20, y=295)
    #
    # Label(window, text="Eps").place(x=150, y=230)
    #
    # eps_entry = Entry(window, width=5)
    # eps_entry.insert(0, 0.5)
    # eps_entry.place(x=150, y=250)
    #
    # Label(window, text="Branching factor").place(x=220, y=230)
    # branching_factor_entry = Entry(window, width=10)
    # branching_factor_entry.insert(0, 50)
    # branching_factor_entry.place(x=220, y=250)
    #
    # Label(window, text="Threeshold").place(x=220, y=275)
    # threeshold = Entry(window, width=10)
    # threeshold.insert(0, 1.5)
    # threeshold.place(x=220, y=295)
    #
    # Label(window, text="Test size").place(x=330, y=230)
    # test_size_entry = Entry(window, width=10)
    # test_size_entry.insert(0, 0.25)
    # test_size_entry.place(x=330, y=250)
    #
    # Label(window, text="Random state").place(x=330, y=275)
    # random_state_entry = Entry(window, width=10)
    # random_state_entry.insert(0, 1)
    # random_state_entry.place(x=330, y=295)

    # log = Text(window)
    # log.place(x=20, y=340)
    # catch_std_out(log)

    def _update_params():
        global params
        # params['nsamples'] = int(nsamples_entry.get())
        # params['cluster_std'] = float(cluster_std_entry.get())
        # params['eps'] = float(eps_entry.get())
        # params['branching_factor'] = int(branching_factor_entry.get())
        # params['threeshold'] = float(threeshold.get())
        # params['test_size'] = float(test_size_entry.get())
        # params['random_state'] = int(random_state_entry.get())

    def _start():
        try:
            _update_params()
            start()
        except Exception as e:
            print("Error: ", e)

    alg = IntVar()

    def _update_ui(a, b, c):
        pass
        # if alg.get() != 0:
        #     eps_entry.configure(state='disabled')
        # else:
        #     eps_entry.configure(state='normal')
        #
        # if alg.get() != 4:
        #     branching_factor_entry.configure(state='disabled')
        #     threeshold.configure(state='disabled')
        # else:
        #     threeshold.configure(state='normal')
        #     branching_factor_entry.configure(state='normal')
        #
        # if alg.get() < 6:
        #     test_size_entry.configure(state='disabled')
        #     random_state_entry.configure(state='disabled')
        #     nsamples_entry.configure(state='normal')
        #     cluster_std_entry.configure(state='normal')
        # else:
        #     test_size_entry.configure(state='normal')
        #     random_state_entry.configure(state='normal')
        #     nsamples_entry.configure(state='disabled')
        #     cluster_std_entry.configure(state='disabled')

    alg.trace_add("write", _update_ui)
    alg.set(0)

    # clust_x = 450
    # Label(window, text="Кластеризация").place(x=clust_x, y=10)
    #
    # Radiobutton(window, text="DB scan", variable=alg, value=0).place(x=clust_x, y=40)
    # Radiobutton(window, text="K means", variable=alg, value=1).place(x=clust_x, y=70)
    # Radiobutton(window, text="Spectral clustering", variable=alg, value=2).place(x=clust_x, y=100)
    # Radiobutton(window, text="Fcm", variable=alg, value=3).place(x=clust_x, y=130)
    # Radiobutton(window, text="Birch", variable=alg, value=4).place(x=clust_x, y=160)
    # Radiobutton(window, text="Agglomerative", variable=alg, value=5).place(x=clust_x, y=190)
    #
    # class_x = 570
    # Label(window, text="Классификация").place(x=class_x, y=10)
    #
    # Radiobutton(window, text="K nearest neighbors", variable=alg, value=6).place(x=class_x, y=40)
    # Radiobutton(window, text="Naive bayes", variable=alg, value=7).place(x=class_x, y=70)
    # Radiobutton(window, text="Decision Tree", variable=alg, value=8).place(x=class_x, y=100)

    # Button(window, text="Начать", command=lambda: _start()).place(x=520, y=230)

    window.mainloop()


def catch_std_out(log):
    sys.stdout = Buffer(log)


def start():
    id = alg.get()
    if id == 0:
        run_db_scan(params)
    elif id == 1:
        run_kmeans(params)
    elif id == 2:
        run_spectral_clustering(params)
    elif id == 3:
        run_fcm(params)
    elif id == 4:
        run_birch(params)
    elif id == 5:
        run_agglomerative(params)
    elif id == 6:
        run_kneares_neighbors(params)
    elif id == 7:
        run_naive_bayes(params)
    elif id == 8:
        run_decision_tree(params)


def add_file_button_clicked(listbox: Listbox):
    filename = filedialog.askopenfilename(initialdir=getcwd())
    listbox.insert(END, filename)


def delete_file_button_clicked(listbox: Listbox):
    index = listbox.curselection()
    if index != ():
        listbox.delete(index)

# TODO: И так для каждого алгоритма
def dbScanTab(frame):
    Label(frame, text="nsamples").place(x=0, y=30)
    nsamples = Entry(frame, width=10)
    nsamples.insert(0, "0.3")
    nsamples.place(x=0, y=50)

    Label(frame, text="Eps").place(x=0, y=80)
    eps = Entry(frame, width=10)
    eps.insert(0, "0.3")
    eps.place(x=0, y=100)

    comboBox = ttk.Combobox(frame,
                                values=[i.value for i in DBSCANMetricType])
    comboBox.current(1)
    comboBox.place(x=0, y=130)

    def _start():
        params['nsamples'] = int(nsamples.get())
        params['eps'] = float(eps.get())
        run_db_scan(params)

    Button(frame, text="Начать", command=lambda: _start()).place(x=0, y=160)

    # self.min_samples = min_samples
    # self.metric = metric
    # self.metric_params = metric_params
    # self.algorithm = algorithm
    # self.leaf_size = leaf_size
    # self.p = p
    # self.n_jobs = n_jobs

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
