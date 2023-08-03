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
    def read_triangle(self, filename=None, sheet_name=None, origin_columns=None, id=None, use_cal=None):
        # replace with passed in values if not None
        
        self.tri = Triangle.from_excel(filename=self.excel_file,
                                       sheet_name=self.tri_sheet_name,
                                       origin_columns=self.origin_columns,
                                       id=self.triangle_id,
                                       use_cal=self.use_cal)
        self.has_tri = True
      
    # 2. read accident period
    def read_acc(self):
        self.acc_df = pd.read_excel(self.excel_file, sheet_name=self.acc_sheet_name)
        self.has_acc = True

    # 3. read development period
    def read_dev(self):
        self.dev_df = pd.read_excel(self.excel_file, sheet_name=self.dev_sheet_name)
        self.has_dev = True

    # 4. read calendar period
    def read_cal(self):
        self.cal_df = pd.read_excel(self.excel_file, sheet_name=self.cal_sheet_name)
        self.has_cal = True

    # 5. read forecast
    def read_forecast(self):
        self.forecast_df = pd.read_excel(self.excel_file, sheet_name=self.forecast_sheet_name)
        self.has_forecast = True

    # 6. read ultimate
    def read_ult(self):
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

