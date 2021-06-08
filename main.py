# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import *
from tkinter import scrolledtext, filedialog

from os import getcwd

from clustering.dbscan.dbscan import DBSCAN
from runner import *

alg = 0


def main():
    global alg

    window = Tk()
    window.title("Text analyzer")
    window.geometry('700x300')

    label = Label(window, text="Файлы")
    label.place(x=10, y=10)

    listbox = Listbox(window, width=40, height=10)
    listbox.place(x=10, y=40)

    add_file_button = Button(window, text="Добавить файл", command=lambda: add_file_button_clicked(listbox))
    add_file_button.place(x=270, y=40)

    delete_file_button = Button(window, text="Удалить файл", command=lambda: delete_file_button_clicked(listbox))
    delete_file_button.place(x=270, y=70)

    alg = IntVar()

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

    Button(window, text="Начать", command=lambda: start(window)).place(x=520, y=190)

    window.mainloop()


def start(window):
    id = alg.get()
    if id == 0:
        run_db_scan()
    elif id == 1:
        run_kmeans()
    elif id == 2:
        run_spectral_clustering()
    elif id == 3:
        run_fcm()
    elif id == 4:
        run_birch()


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
