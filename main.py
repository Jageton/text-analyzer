# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import *
from tkinter import scrolledtext, filedialog

from os import getcwd

from buffer import Buffer
from runner import *

alg = 0

params = {}


def main():
    global alg

    window = Tk()
    window.title("Text analyzer")
    window.geometry('700x740')

    label = Label(window, text="Файлы")
    label.place(x=10, y=10)

    listbox = Listbox(window, width=40, height=10)
    listbox.place(x=10, y=40)

    add_file_button = Button(window, text="Добавить файл", command=lambda: add_file_button_clicked(listbox))
    add_file_button.place(x=270, y=40)

    delete_file_button = Button(window, text="Удалить файл", command=lambda: delete_file_button_clicked(listbox))
    delete_file_button.place(x=270, y=70)

    Label(window, text="Count of points").place(x=20, y=230)

    nsamples_entry = Entry(window, width=10)
    nsamples_entry.insert(0, "1000")
    nsamples_entry.place(x=20, y=250)

    Label(window, text="Cluster std").place(x=20, y=275)

    cluster_std_entry = Entry(window, width=10)
    cluster_std_entry.insert(0, 1)
    cluster_std_entry.place(x=20, y=295)

    Label(window, text="Eps").place(x=150, y=230)

    eps_entry = Entry(window, width=5)
    eps_entry.insert(0, 0.5)
    eps_entry.place(x=150, y=250)

    Label(window, text="Branching factor").place(x=220, y=230)
    branching_factor_entry = Entry(window, width=10)
    branching_factor_entry.insert(0, 50)
    branching_factor_entry.place(x=220, y=250)

    Label(window, text="Threeshold").place(x=220, y=275)
    threeshold = Entry(window, width=10)
    threeshold.insert(0, 1.5)
    threeshold.place(x=220, y=295)

    log = Text(window)
    log.place(x=20, y=340)
    catch_std_out(log)

    def _update_params():
        global params
        params['nsamples'] = int(nsamples_entry.get())
        params['cluster_std'] = float(cluster_std_entry.get())
        params['eps'] = float(eps_entry.get())
        params['branching_factor'] = int(branching_factor_entry.get())
        params['threeshold'] = float(threeshold.get())

    def _start():
        try:
            _update_params()
            start()
        except Exception as e:
            print("Error: ", e)

    alg = IntVar()

    def _update_ui(a, b, c):
        if alg.get() != 0:
            eps_entry.configure(state='disabled')
        else:
            eps_entry.configure(state='normal')

        if alg.get() != 4:
            branching_factor_entry.configure(state='disabled')
            threeshold.configure(state='disabled')
        else:
            threeshold.configure(state='normal')
            branching_factor_entry.configure(state='normal')

    alg.trace_add("write", _update_ui)
    alg.set(0)

    clust_x = 450
    Label(window, text="Кластеризация").place(x=clust_x, y=10)

    Radiobutton(window, text="DB scan", variable=alg, value=0).place(x=clust_x, y=40)
    Radiobutton(window, text="K means", variable=alg, value=1).place(x=clust_x, y=70)
    Radiobutton(window, text="Spectral clustering", variable=alg, value=2).place(x=clust_x, y=100)
    Radiobutton(window, text="Fcm", variable=alg, value=3).place(x=clust_x, y=130)
    Radiobutton(window, text="Birch", variable=alg, value=4).place(x=clust_x, y=160)

    class_x = 570
    Label(window, text="Классификация").place(x=class_x, y=10)

    Radiobutton(window, text="K nearest neighbors", variable=alg, value=5).place(x=class_x, y=40)
    Radiobutton(window, text="Naive bayes", variable=alg, value=6).place(x=class_x, y=70)

    Button(window, text="Начать", command=lambda: _start()).place(x=520, y=190)

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


def add_file_button_clicked(listbox: Listbox):
    filename = filedialog.askopenfilename(initialdir=getcwd())
    listbox.insert(END, filename)


def delete_file_button_clicked(listbox: Listbox):
    index = listbox.curselection()
    if index != ():
        listbox.delete(index)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
