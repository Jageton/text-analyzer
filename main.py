# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import urllib.request
from enum import Enum
from tkinter import *
from tkinter import ttk, filedialog, messagebox

from os import getcwd
from tkinter.ttk import Notebook

import pandas

from buffer import Buffer
from clustering.dbscan.dbscan import DBSCANMetricType, DBSCANAlgorithmType
from runner import *
from pio.input import Input

listbox = None


def main():
    window = Tk()
    window.title("Text analyzer")
    window.geometry('700x740')
    tabControl = Notebook(window, height=440, width=700)

    tab1 = ttk.Frame(tabControl)
    tab2 = ttk.Frame(tabControl)
    log_tab = ttk.Frame(tabControl)

    tabControl.add(tab1, text='DB SCAN')
    tabControl.add(tab2, text='Birch')
    tabControl.add(log_tab, text='Log')
    tabControl.place(x=0, y=300)

    db_scan_tab(tab1)
    fill_log_tab(log_tab)

    ttk.Label(tab2,
              text="Lets dive into the\
              world of computers").grid(column=0,
                                        row=0,
                                        padx=30,
                                        pady=30)

    label = Label(window, text="Ресурсы")
    label.place(x=10, y=10)

    global listbox
    listbox = Listbox(window, width=40, height=10)
    listbox.place(x=10, y=40)

    add_file_button = Button(window, text="Добавить файл", command=lambda: add_file_button_clicked(listbox))
    add_file_button.place(x=370, y=40)

    delete_file_button = Button(window, text="Удалить", command=lambda: delete_file_button_clicked(listbox))
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
                    messagebox.showinfo(title='Проверка подклчения', message='URL существует')
                else:
                    messagebox.showerror(title='Проверка подклчения', message='URL не существует')
            except ValueError:
                messagebox.showerror(title='Проверка подклчения', message='URL не существует')
            except Exception:
                messagebox.showerror(title='Проверка подклчения', message='URL не существует')

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


# TODO: И так для каждого алгоритма
def db_scan_tab(frame):
    eps = add_input(10, 30, frame, "Eps", 0.5)
    metric = add_input(10, 80, frame, "Metric", DBSCANMetricType)
    algorithm = add_input(10, 130, frame, "Algorithm", DBSCANAlgorithmType)
    leaf_size = add_input(150, 30, frame, "Leaf size", 30)
    p = add_input(150, 80, frame, "Minkowski metric power", "2 or None")
    nsamples = add_input(10, 290, frame, "Nsamples", 10000)

    def _start_with_nsamples():
        power = 2
        if p.get() != "2":
            power = None
        run_db_scan(nsamples=int(nsamples.get()), dataframe=None, eps=float(eps.get()), metric=metric.get(),
                    algorithm=algorithm.get(),
                    leaf_size=int(leaf_size.get()),
                    p=power)

    def _start_with_file():
        try:
            path = listbox.get(ACTIVE)
            if path != "":
                dataframe = pandas.read_csv(path)

                power = 2
                if p.get() != "2":
                    power = None

                def _start(df):
                    return lambda: DBSCAN(eps=float(eps.get()), metric=metric.get(),
                                          algorithm=algorithm.get(),
                                          leaf_size=int(leaf_size.get()),
                                          p=power).run(df)

                run_algorithm(frame, dataframe.copy(), _start)
            else:
                messagebox.showerror(message='Не был выбран файл или URL')
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            messagebox.showerror(message="Произошла ошибочка. Посмотрите в лог.")

    Button(frame, text="Запустить с nsamples", command=_start_with_nsamples).place(x=10, y=340)
    Button(frame, text="Запустить на файле", command=_start_with_file).place(x=200, y=340)


def run_algorithm(frame, dataframe, algorithm):
    new_window = Toplevel(frame)
    new_window.columnconfigure(0, pad=10)
    new_window.columnconfigure(1, pad=10)
    new_window.columnconfigure(2, pad=10)
    new_window.rowconfigure(0, pad=10)
    new_window.rowconfigure(1, pad=10)
    new_window.rowconfigure(2, pad=20)

    del dataframe[dataframe.columns[-1]]

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

        alg = algorithm(df)
        if len(df.columns) == 2:
            run_algorithm_for_2_columns(alg, df)
        else:
            run_algorithm_for_3_columns(alg, df)

    Button(new_window, text='Запустить', command=_start).grid(column=1, row=2)


def add_input(x, y, frame, text, value):
    Label(frame, text=text).place(x=x, y=y)
    if isinstance(value, (str, int, float)):
        entry = Entry(frame, width=10)
        entry.insert(0, value)
        entry.place(x=x, y=y + 20)
        return entry
    elif issubclass(value, Enum):
        combo_box = ttk.Combobox(frame,
                                 values=[i.value for i in value], width=10)
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
    else:
        if path.endswith('csv'):
            return Input.local_read_csv(path)
        return Input.local_read_text_file(path)


def fill_log_tab(log_tab):
    log = Text(log_tab, width=87, height=26)
    log.place(x=0, y=0)
    catch_std_out(log)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
