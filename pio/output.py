class Output:
    @staticmethod
    def write_to_txt_file(file_path, file):
        """
        Записывает в txt файл
        :param file_path: путь файла для записи
        :param file: записываемый файл
        """

        try:
            with open(file_path, 'w', encoding='utf-8') as file2:
                file2.write(str(file))
        except TimeoutError:
            print('Истекло время ожидания')
        except Exception:
            print('Неизвестная ошибка!')

    @staticmethod
    def write_to_csv_file(file_path, csv_table):
        """
        Записывает в csv файл
        :param file_path: путь файла для записи
        :param csv_table: записываемая csv таблица
        """

        try:
            csv_table.to_csv(file_path)
        except TimeoutError:
            print('Истекло время ожидания')
        except Exception:
            print('Неизвестная ошибка!')

    @staticmethod
    def write_array_to_txt_file(file_path, array):
        """
        Записывает массив в файл
        :param file_path: путь файла для записи
        :param array: записываемый массив
        """

        try:
            with open(file_path, 'w', encoding='utf-8') as file2:
                for line in array:
                    file2.write(str(line) + "\n")
        except TimeoutError:
            print('Истекло время ожидания')
        except Exception:
            print('Неизвестная ошибка!')
