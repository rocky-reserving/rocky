import pandas as pd
import re
from openpyxl.utils.cell import column_index_from_string

def formatter(x, kind: str = "dollars") -> str:
    if kind=="dollars":
        return '$ {:0,.0f}'.format(x)
    elif kind=='factors':
        return '{:.3f}'.format(x)
    elif kind=='percents':
        return '{:.1f}%'.format(x*100)
    
def get_allowed_triangle_types() -> list:
    allowed_triangle_types = [
        'paid_loss',
        'reported_loss',
        'paid_dcce',
        'paid_loss_dcce',
        'reported_loss_dcce',
        'case_reserves',
        'reported_counts',
        'closed_counts',
        'open_counts']
    return allowed_triangle_types



def _read_excel_range(filename : str = None,
                      sheetname : str = None,
                      range_address : str = None
                      ) -> pd.DataFrame:
    """
    Read a range of cells from an Excel file into a Pandas DataFrame.

    Parameters
    ----------
    filename : str
        The path to the Excel file.
    sheetname : str
        The name of the sheet in the Excel file.
    range_address : str
        The range of cells to read, e.g. "A1:C5".

    Returns
    -------
    df : pd.DataFrame
        A Pandas DataFrame containing the data in the specified range.
    """
    # get the range as a tuple of tuples
    # Parse the range address to get the starting and ending row and column indices
    pattern = re.compile(r"([A-Z]+)(\d+):([A-Z]+)(\d+)")
    match = pattern.match(range_address)
    c1, r1, c2, r2 = match.groups()
    r1 = int(r1)
    r2 = int(r2)
    c1 = column_index_from_string(c1)
    c2 = column_index_from_string(c2)

    # Load the data in the specified range into a Pandas DataFrame
    df = pd.read_excel(filename, sheetname, header=None, index_col=None)

    # Slice the DataFrame to the specified range
    df = df.iloc[(r1 - 1):r2, (c1 - 1):c2]

    # Set the column names to the values in the first row
    df.columns = df.iloc[0]
    df = df[1:]

    return df
