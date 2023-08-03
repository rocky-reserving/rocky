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

    def __repr__(self):
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

        has_any_attr = len(attributes) > 0

        attr_string = ', '.join(attributes)

        if has_any_attr:
            return f"adrian(inputs={attr_string})"
        else:
            return f"adrian(inputs='{self.excel_file}')"

    # methods
    # 1. read triangle
    # 2. read accident period
    # 3. read development period
    # 4. read calendar period
    # 5. read forecast
    # 6. read ultimate
    # 7. read all
    # 8. write empty triangle (to excel)
    # 9. write empty accident period (to excel)
    # 10. write empty development period (to excel)
    # 11. write empty calendar period (to excel)
    # 12. write empty forecast (to excel)
    # 13. write empty ultimate (to excel)

