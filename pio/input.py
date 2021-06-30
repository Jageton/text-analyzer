import pandas
import requests as requests


class Input:

    @staticmethod
    def internet_read_csv(url):
        """
        Чтение csv таблиц из интернета
        :param url: адрес файла в интернете
        :return: csv таблица
        """

        print('Чтение CSV с ' + url + '...')
        try:
            return pandas.read_csv(url)
        except TimeoutError:
            print('Истекло время ожидания')
        except Exception:
            print('Неизвестная ошибка!')

    @staticmethod
    def local_read_csv(file_path):
        """
        Чтение csv таблиц из файла
        :param file_path: путь к файлу
        :return: csv таблица
        """

        print('Чтение CSV из ' + file_path + '...')
        try:
            return pandas.read_csv(file_path, index_col=None)
        except TimeoutError:
            print('Истекло время ожидания')
        except Exception:
            print('Неизвестная ошибка!')

    @staticmethod
    def local_read_text_file(file_path):
        """
        Чтение текстовых файлов с компьютера
        :param file_path: путь к файлу
        :return: текст
        """

        print('Чтение данных из ' + file_path)
        try:
            file2 = ""
            with open(file_path, 'r', encoding='utf-8') as file1:
                for line in file1:
                    file2 = file2 + str(line)
            return file2
        except TimeoutError:
            print('Истекло время ожидания')
        except Exception:
            print('Неизвестная ошибка')

    @staticmethod
    def internet_read_text_file(url):
        """
        Чтение текстовых файлов из интернета
        :param url: адрес файла в интернете
        :return: текст
        """

        print('Чтение данных с ' + url)
        try:
            return requests.get(url).text
        except TimeoutError:
            print('Истекло время ожидания')
        except Exception:
            print('Неизвестная ошибка!')
