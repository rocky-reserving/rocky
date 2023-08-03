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
    def read_inputs(self):
        self.read_triangle()
        self.read_acc()
        self.read_dev()
        self.read_cal()
        self.read_forecast()
        self.read_ult()

    # 8. write empty triangle (to excel)
    # 9. write empty accident period (to excel)
    # 10. write empty development period (to excel)
    # 11. write empty calendar period (to excel)
    # 12. write empty forecast (to excel)
    # 13. write empty ultimate (to excel)

