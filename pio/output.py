import openpyxl
from openpyxl.styles import Alignment
from openpyxl.utils.dataframe import dataframe_to_rows


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

    @staticmethod
    def write_to_xlsx_file(file_path, data_frame, clusters_dict):
        """
        Записывает массив в файл
        :param file_path: путь файла для записи
        :param data_frame: исходные данные
        :param clusters_dict: распределение по кластерам
        """

        workbook = openpyxl.Workbook()
        ws_source_data = workbook.active
        ws_source_data.title = 'Исходные данные'

        rows = dataframe_to_rows(df=data_frame, index=False)
        for r_idx, row in enumerate(rows, 1):
            for c_idx, value in enumerate(row, 1):
                cell = ws_source_data.cell(row=r_idx, column=c_idx, value=value)
                cell.alignment = Alignment(horizontal='center')

        Output._format_sheet(ws_source_data)

        ws_result = workbook.create_sheet('Результат кластеризации')

        for c_idx, (key, values) in enumerate(clusters_dict.items(), 1):
            cell = ws_result.cell(row=1, column=c_idx, value=key)
            cell.alignment = Alignment(horizontal='center')
            for r_idx, value in enumerate(values, 1):
                cell = ws_result.cell(row=r_idx + 1, column=c_idx, value=Output._points_to_str(value))
                cell.alignment = Alignment(horizontal='center')

        Output._format_sheet(ws_result)

        workbook.save(file_path)
        workbook.close()

    @staticmethod
    def _points_to_str(list):
        return '(%s)' % ';'.join(map(lambda x: str(x), list))

    @staticmethod
    def _format_sheet(worksheet):
        for column_cells in worksheet.columns:
            length = max(len(str(cell.value or "")) for cell in column_cells) + 2
            worksheet.column_dimensions[column_cells[0].column_letter].width = length
