import pandas as pd

from dataclasses import dataclass

from .triangle import Triangle

@dataclass
class adrian:
    excel_file: str = "./rocky-inputs.xslx"
    tri_sheet_name: str = "triangle"
    acc_sheet_name: str = "accident_period"
    dev_sheet_name: str = "development_period"
    cal_sheet_name: str = "calendar_period"
    forecast_sheet_name: str = "forecast"
    ult_sheet_name: str = "ultimate"

    tri: Triangle = None
    has_tri: bool = False
    
    acc_df: pd.DataFrame = None
    has_acc: bool = False

    dev_df: pd.DataFrame = None
    has_dev: bool = False

    cal_df: pd.DataFrame = None
    has_cal: bool = False

    forecast_df: pd.DataFrame = None
    has_forecast: bool = False

    ult_df: pd.DataFrame = None
    has_ult: bool = False

    origin_columns: int = 1
    triangle_id: str = "loss"
    use_cal: bool = True

    def __repr__(self):
        # build a list of attributes that are set
        attributes = []
        if self.has_tri:
            attributes.append('triangle')
        if self.has_acc:
            attributes.append('acc')
        if self.has_dev:
            attributes.append('dev')
        if self.has_cal:
            attributes.append('cal')
        if self.has_forecast:
            attributes.append('forecast')
        if self.has_ult:
            attributes.append('ult')

        # build a string of attributes that are set
        attr_string = ', '.join(attributes)

        # different repr if no attributes are set
        has_any_attr = len(attributes) > 0

        # return repr
        if has_any_attr:
            return f"adrian(loaded_data=({attr_string}))"
        else:
            return f"adrian(inputs='{self.excel_file}')"

    # methods
    # 1. read triangle
    def read_triangle(self,
                      filename:str = None,
                      sheet_name:str = None,
                      origin_columns:int = None,
                      id:str = None,
                      use_cal:bool = None) -> None:
        """
        Reads a triangle from an excel file and sets the triangle attribute.

        ### Parameters
        
        `filename` : `str`, optional
            The name of the excel file to read from. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to read from. The default is None.
        `origin_columns` : int, optional
            The number of columns to skip before the origin columns. The default is None.
        `id` : str, optional
            The id of the triangle. The default is None.
        `use_cal` : bool, optional
            Whether to use the calendar period. The default is None.

        ### Returns
        
        `None`. Sets the `triangle` attribute.

        ### Examples
        
        >>> a = adrian(filename='triangle.xlsx',
                       sheet_name='triangle',
                       origin_columns=1,
                       id='loss',
                       use_cal=True)
        >>> # read in triangle from excel file
        >>> a.read_triangle()
        
        >>> # read in triangle from excel file with different parameters
        >>> a.read_triangle(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)
        """

        # replace with passed in values if not None
        if filename is not None:
            self.excel_file = filename
        if sheet_name is not None:
            self.tri_sheet_name = sheet_name
        if origin_columns is not None:
            self.origin_columns = origin_columns
        if id is not None:
            self.triangle_id = id
        if use_cal is not None:
            self.use_cal = use_cal

        # read triangle & set has_tri to True
        self.tri = Triangle.from_excel(filename=self.excel_file,
                                       sheet_name=self.tri_sheet_name,
                                       origin_columns=self.origin_columns,
                                       id=self.triangle_id,
                                       use_cal=self.use_cal)
        self.has_tri = True
      
    # 2. read accident period
    def read_acc(self,
                 filename:str = None,
                 sheet_name:str = None) -> None:
        """
        Reads the accident period from an excel file and sets the `acc_df` attribute.

        ### Parameters

        `filename` : `str`, optional
            The name of the excel file to read from. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to read from. The default is None.
        
        ### Returns

        `None`. Sets the `acc_df` attribute.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                          sheet_name='triangle',
                          origin_columns=1,
                          id='loss',
                          use_cal=True)
        
        >>> # read in accident period from excel file
        >>> a.read_acc()

        >>> # read in accident period from a different excel file
        >>> a.read_acc(filename='accident_period.xlsx',
                       sheet_name='accident_period')
        """
        # replace with passed in values if not None
        if filename is not None:
            self.excel_file = filename
        if sheet_name is not None:
            self.acc_sheet_name = sheet_name

        # read accident period & set has_acc to True
        self.acc_df = pd.read_excel(self.excel_file, sheet_name=self.acc_sheet_name)
        self.has_acc = True

    # 3. read development period
    def read_dev(self,
                 filename:str = None,
                 sheet_name:str = None) -> None:
        """
        Reads the development period from an excel file and sets the `dev_df` attribute.

        ### Parameters

        `filename` : `str`, optional
            The name of the excel file to read from. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to read from. The default is None.

        ### Returns

        `None`. Sets the `dev_df` attribute.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)

        >>> # read in development period from excel file
        >>> a.read_dev()

        >>> # read in development period from a different excel file
        >>> a.read_dev(filename='development_period.xlsx',
                          sheet_name='development_period')
          """
        # replace with passed in values if not None
        if filename is not None:
            self.excel_file = filename
        if sheet_name is not None:
            self.dev_sheet_name = sheet_name

        # read development period & set has_dev to True
        self.dev_df = pd.read_excel(filename=self.excel_file,
                                    sheet_name=self.dev_sheet_name)
        self.has_dev = True

    # 4. read calendar period
    def read_cal(self,
                 filename:str = None,
                 sheet_name:str = None) -> None:
        """
        Reads the calendar period from an excel file and sets the `cal_df` attribute.

        ### Parameters

        `filename` : `str`, optional
            The name of the excel file to read from. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to read from. The default is None.

        ### Returns

        `None`. Sets the `cal_df` attribute.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)

        >>> # read in calendar period from excel file
        >>> a.read_cal()

        >>> # read in calendar period from a different excel file
        >>> a.read_cal(filename='calendar_period.xlsx',
                          sheet_name='calendar_period')
        """
        self.cal_df = pd.read_excel(filename=self.excel_file,
                                    sheet_name=self.cal_sheet_name)
        self.has_cal = True

    # 5. read forecast
    def read_forecast(self,
                      filename:str = None,
                      sheet_name:str = None) -> None:
        """
        Reads the forecast from an excel file and sets the `forecast_df` attribute.

        ### Parameters

        `filename` : `str`, optional
            The name of the excel file to read from. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to read from. The default is None.

        ### Returns

        `None`. Sets the `forecast_df` attribute.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)

        >>> # read in forecast from excel file
        >>> a.read_forecast()

        >>> # read in forecast from a different excel file
        >>> a.read_forecast(filename='forecast.xlsx',
                            sheet_name='forecast')
        """
        # replace with passed in values if not None
        if filename is not None:
            self.excel_file = filename
        if sheet_name is not None:
            self.forecast_sheet_name = sheet_name

        # read forecast & set has_forecast to True
        self.forecast_df = pd.read_excel(filename=self.excel_file,
                                         sheet_name=self.forecast_sheet_name)
        self.has_forecast = True

    # 6. read ultimate
    def read_ult(self, 
                 filename:str = None,
                 sheet_name:str = None) -> None:
        """
        Reads the ultimate from an excel file and sets the `ult_df` attribute.

        ### Parameters

        `filename` : `str`, optional
            The name of the excel file to read from. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to read from. The default is None.

        ### Returns

        `None`. Sets the `ult_df` attribute.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)

        >>> # read in ultimate from excel file
        >>> a.read_ult()

        >>> # read in ultimate from a different excel file
        >>> a.read_ult(filename='ultimate.xlsx',
                            sheet_name='ultimate')
        """
        # replace with passed in values if not None
        if filename is not None:
            self.excel_file = filename
        if sheet_name is not None:
            self.ult_sheet_name = sheet_name

        # read ultimate & set has_ult to True
        self.ult_df = pd.read_excel(self.excel_file, sheet_name=self.ult_sheet_name)
        self.has_ult = True

    # 7. read all
    def read_inputs(self,
                    filename:str = None,
                    sheet_name:str = None,
                    origin_columns:int = None,
                    id:str = None,
                    use_cal:bool = None) -> None:
        """
        Reads all the inputs from an excel file and sets the corresponding attributes.

        ### Parameters
        filename : `str`, optional
            The name of the excel file to read from. The default is None.
        sheet_name : `str`, optional
            The name of the sheet to read from. The default is None.
        origin_columns : `int`, optional
            The number of columns to skip before the origin columns. The default is None.
        id : `str`, optional
            The name of the column that contains the id. The default is None.
        use_cal : `bool`, optional
            Whether to use calendar period. The default is None.

        ### Returns

        `None`. Sets the corresponding attributes.
        """
        # replace with passed in values if not None
        if filename is not None:
            self.excel_file = filename
        if sheet_name is not None:
            self.triangle_sheet_name = sheet_name
        if origin_columns is not None:
            self.origin_columns = origin_columns
        if id is not None:
            self.id = id
        if use_cal is not None:
            self.use_cal = use_cal

        # read all inputs
        self.read_triangle()
        self.read_acc()
        self.read_dev()
        self.read_cal()
        self.read_forecast()
        self.read_ult()

    def _calculate_triangle_row_col(self,
                                    start_acc:int = None,
                                    end_acc:int = None,
                                    n_dev: int = None
                                   ) -> tuple[int, int, int]:
        """
        Calculates the row and column of the triangle to be read. Helper function
        that is not meant to be called directly.

        ### Parameters

        `start_acc` : `int`, optional
            The starting accident period to read from. The default is None.
        `end_acc` : `int`, optional
            The ending accident period to read from. The default is None.
        `n_dev` : `int`, optional
            The number of development periods to read. The default is None.

        ### Returns

        `tuple[int, int, int]`. The row and column of the triangle to be read.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)

        >>> s = 1
        >>> d = 10
        >>> # expect: e = d - s + 1 = 10 - 1 + 1 = 10
        >>> a._calculate_triangle_row_col(start_acc=1,
                                          end_acc=10,
                                          n_dev=None)
        (1, 10, 10)

        >>> s = 1
        >>> e = 10
        >>> # expect: d = e - s + 1 = 10 - 1 + 1 = 10
        >>> a._calculate_triangle_row_col(start_acc=1,
                                          end_acc=10,
                                          n_dev=None)
        (1, 10, 10)

        >>> e = 20
        >>> d = 10
        >>> # expect: s = e - d + 1 = 20 - 10 + 1 = 11
        >>> a._calculate_triangle_row_col(start_acc=None,
                                          end_acc=20,
                                          n_dev=10)
        (11, 20, 10)

        >>> s = None
        >>> e = None
        >>> d = 10
        >>> # should throw error since 2/3 of s, e, d must be specified
        >>> a._calculate_triangle_row_col(start_acc=None,
                                            end_acc=None,
                                            n_dev=10)
        Traceback (most recent call last):
        ...
        ValueError: 2/3 of start_acc, end_acc, n_dev must be specified
        """
        # start_acc, end_acc, n_dev
        # start_acc, end_acc, None
        # start_acc, None, n_dev
        # None, end_acc, n_dev
        # start_acc, None, None
        # None, end_acc, None
        # None, None, n_dev
        # None, None, None

        # 2/3 of start_acc, end_acc, n_dev must be specified
        if ((start_acc is None) & (end_acc is None) & (n_dev is None)):
            raise ValueError('2/3 of start_acc, end_acc, n_dev must be specified')

        # start_acc, end_acc, n_dev
        if start_acc is not None:
            if end_acc is not None:
                if n_dev is not None:
                    assert n_dev == end_acc - start_acc + 1, \
                    f"""n_dev must equal end_acc - start_acc + 1
                    n_dev: {n_dev}
                    end_acc: {end_acc}
                    start_acc: {start_acc}"""
                    return start_acc, end_acc, n_dev
                else:
                    return start_acc, end_acc, end_acc - start_acc + 1
            else:
                end_acc = start_acc + n_dev - 1
                return start_acc, end_acc, n_dev
        else:
            if end_acc is not None:
                if n_dev is not None:
                    assert n_dev == end_acc - start_acc + 1, \
                    f"""n_dev must equal end_acc - start_acc + 1
                    n_dev: {n_dev}
                    end_acc: {end_acc}
                    start_acc: {start_acc}"""
                    start_acc = end_acc - n_dev + 1
                    return start_acc, end_acc, n_dev
                else:
                    n_dev = end_acc - start_acc + 1
                    return start_acc, end_acc, n_dev
            else:
                start_acc = end_acc - n_dev + 1
                return start_acc, end_acc, n_dev
                

    # 8. write empty triangle (to excel)
    def write_triangle(self,
                       filename:str = None,
                       sheet_name:str = None,
                       start_accident_period:int = None,
                       end_accident_period:int = None,
                       n_development_periods:int = None,
                       ) -> None:
        """
        Writes an empty triangle to an excel file. This triangle can be used to 
        input data into the triangle to load into the `adrian` object.

        ### Parameters

        `filename` : `str`, optional
            The name of the excel file to write to. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to write to. The default is None.
        `start_accident_period` : int, optional
            The start of the accident period. The default is None, but
            if `end_accident_period` is not None and `n_development_periods`
            is not None, then `start_accident_period` is calculated as
            `end_accident_period` - `n_development_periods` + 1.
        `end_accident_period` : int, optional
            The end of the accident period. The default is None, but if
            `start_accident_period` is not None and `n_development_periods`
            is not None, then `end_accident_period` is calculated as
            `start_accident_period` + `n_development_periods` - 1.
        `n_development_periods` : int, optional
            The number of development periods. The default is None, but 
            if both `start_accident_period` and `end_accident_period` are
            not None, then `n_development_periods` is calculated as
            `end_accident_period` - `start_accident_period` + 1.

        ### Returns

        `None`. Writes an empty triangle to an excel file.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)

        >>> # write empty triangle to the same excel file
        >>> # (note that this will overwrite the existing triangle)
        >>> a.write_triangle()

        >>> # write empty triangle to a different excel file
        >>> # (will create the file/tab if it does not exist)
        >>> a.write_triangle(filename='empty_triangle.xlsx',
                            sheet_name='empty_triangle')
        """
        # replace with passed in values if not None
        if filename is not None:
            self.excel_file = filename
        if sheet_name is not None:
            self.triangle_sheet_name = sheet_name

        # calculate start_accident_period, end_accident_period, n_development_periods
        start_accident_period, end_accident_period, n_development_periods = \
            self._calculate_triangle_row_col(start_accident_period,
                                             end_accident_period,
                                             n_development_periods)

        # index ranges from start_accident_period to end_accident_period
        idx = range(start_accident_period, end_accident_period + 1)

        # columns range from 1 to n_development_periods
        cols = range(1, n_development_periods + 1)

        # create empty dataframe
        blank_df = pd.DataFrame(index=idx, columns=cols)

        # write to excel
        blank_df.to_excel(self.excel_file,
                            sheet_name=self.triangle_sheet_name,
                            index=True,
                            header=True)

    # 9. write empty accident period (to excel)
    def write_accident_period(self,
                              filename:str = None,
                              sheet_name:str = None,
                              start_accident_period:int = None,
                              end_accident_period:int = None,
                              n_development_periods:int = None,
                              ) -> None:
        """
        Writes an empty accident period to an excel file. This accident period
        can be used to input data into the triangle to load into the `adrian`
        object.

        ### Parameters

        `filename` : `str`, optional
            The name of the excel file to write to. The default is None.
        `sheet_name` : str, optional
            The name of the sheet to write to. The default is None.
        `start_accident_period` : int, optional
            The start of the accident period. The default is None, but
            if `end_accident_period` is not None and `n_development_periods`
            is not None, then `start_accident_period` is calculated as
            `end_accident_period` - `n_development_periods` + 1.
        `end_accident_period` : int, optional
            The end of the accident period. The default is None, but if
            `start_accident_period` is not None and `n_development_periods`
            is not None, then `end_accident_period` is calculated as
            `start_accident_period` + `n_development_periods` - 1.
        `n_development_periods` : int, optional
            The number of development periods. The default is None, but 
            if both `start_accident_period` and `end_accident_period` are
            not None, then `n_development_periods` is calculated as
            `end_accident_period` - `start_accident_period` + 1.

        ### Returns

        `None`. Writes an empty accident period to an excel file.

        ### Examples

        >>> a = adrian(filename='triangle.xlsx',
                            sheet_name='triangle',
                            origin_columns=1,
                            id='loss',
                            use_cal=True)

        >>> # write empty accident period to the same excel file
        >>> # (note that this will overwrite the existing accident period)
        >>> a.write_accident_period()

        >>> # write empty accident period to a different excel file
        >>> # (will create the file/tab if it does not exist)
        >>> a.write_accident_period(filename='empty_accident_period.xlsx',
                                   sheet_name='empty_accident_period')
        """

    # 10. write empty development period (to excel)
    # 11. write empty calendar period (to excel)
    # 12. write empty forecast (to excel)
    # 13. write empty ultimate (to excel)

