"""
This module implements the Triangle class, which is used to store and manipulate triangle data.

This class also includes methods for perfoming basic loss triangle analysis using the chain ladder method.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Union, Optional, Tuple

from openpyxl.utils import range_to_tuple

from utils import formatter, get_allowed_triangle_types, _read_excel_range
# from chainladder import ChainLadder

# get the aliases for the triangle types
triangle_type_aliases = get_allowed_triangle_types()


@dataclass
class Triangle:
    """
    Create a `Triangle` object. The `Triangle` object is used to store and manipulate triangle data.
    Attributes:
    -----------
    id : str
        The type of triangle the object represents - paid loss, reported loss, etc.
    tri : pd.DataFrame, default=None
        The triangle data. Must be a pandas DataFrame with:
            1. The origin period set as the index.
            2. The development periods set as the column names.
            3. The values set as the values in the DataFrame.
        If any of these conditions are not met, the triangle data will be set to None.
    """
    id: str = None
    tri: pd.DataFrame = None
    triangle: pd.DataFrame = None
    incr_triangle: pd.DataFrame = None
    X_base: pd.DataFrame = None
    y_base: np.ndarray = None
    X_base_train: pd.DataFrame = None
    y_base_train: np.ndarray = None
    X_base_forecast: pd.DataFrame = None
    y_base_forecast: np.ndarray = None

    def __post_init__(self) -> None:
        """
        Reset triangle id if it is not allowed.
        Parameters:
        -----------
        None
        Returns:
        --------
        None
        """
        # reformat the id
        self.id = self.id.lower().replace(" ", "_")

        # reset the id if it is not allowed
        if self.id not in triangle_type_aliases:
            self.id = None

        # if a triangle was passed in, set the n_rows and n_cols attributes
        if self.tri is not None:
            self.n_rows = self.tri.shape[0]
            self.n_cols = self.tri.shape[1]

        

    def __repr__(self) -> str:
        return self.tri.__repr__()
    
    def __str__(self) -> str:
        return self.tri.__str__()
    
    def set_id(self, id: str) -> None:
        """
        Set the id of the triangle.
        Parameters:
        -----------
        `id`: `str`
            The id of the triangle.
        Returns:
        --------
        `None`
        """
        # ensure that the id is a string
        if not isinstance(id, str):
            raise TypeError('The id must be a string.')
        
        # ensure the id is allowed
        if id.lower().replace(" ", "_") in triangle_type_aliases:
            self.id = id.lower().replace(" ", "_")
        else:
            print(f'The id {id} is not allowed. It must be one of the following:')
            for alias in triangle_type_aliases:
                print(f"  - {alias}")
            print()
            raise ValueError('The id is not allowed.')
        
    @classmethod
    def from_dataframe(cls,
                       id: str,
                       df: pd.DataFrame
                       ) -> 'Triangle':
        """
        Create a Triangle object from a pandas DataFrame.
        
        Parameters:
        -----------
        id : str
            The id of the triangle.
        df : pd.DataFrame
            The triangle data. Must be a pandas DataFrame with:
                1. The origin period set as the index.
                2. The development periods set as the column names.
                3. The values set as the values in the DataFrame.
            If any of these conditions are not met, the triangle data will be set to None.

        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the DataFrame.

        """
        # Create and return a Triangle object
        return cls(id=id, tri=df, triangle=df)
        
    @classmethod
    def from_clipboard(cls,
                       id: str,
                       origin_columns: int
                       ) -> 'Triangle':
        """
        Create a Triangle object from data copied to the clipboard.
        Parameters:
        -----------
        id : str
            The id of the triangle.
        origin_columns : int
            The number of columns used for the origin period.
        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the clipboard.
        """
        # Read data from the clipboard, assuming the first row is the development period
        # and the first `origin_columns` columns should make up either an index or a
        # multi-index for the origin period in the resulting DataFrame
        df = pd.read_clipboard(header=0, index_col=range(origin_columns))

        # Create and return a Triangle object
        return cls(id=id, tri=df, triangle=df)

    
    @classmethod
    def from_csv(cls,
                 filename: str,
                 id: str,
                 origin_columns: int
                 ) -> 'Triangle':
        """
        Create a Triangle object from data in a CSV file.
        Parameters:
        -----------
        filename : str
            The name of the CSV file containing the triangle data.
        id : str
            The id of the triangle.
        origin_columns : int
            The number of columns used for the origin period.
        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the CSV file.
        """
        # Read data from the CSV file
        df = pd.read_csv(filename, header=0, index_col=[i for i in range(origin_columns)])

        # Create and return a Triangle object
        return cls(id=id, tri=df, triangle=df)
    
    @classmethod
    def from_excel(cls,
                   filename: str,
                   id: str,
                   origin_columns: int,
                   sheet_name: Optional[str] = None,
                   sheet_range: Optional[str] = None
                   ) -> 'Triangle':
        """
        Create a Triangle object from data in an Excel file.
        Parameters:
        -----------
        filename : str
            The name of the Excel file containing the triangle data.
        id : str
            The id of the triangle.
        origin_columns : int
            The number of columns used for the origin period.
        sheet_name : str, optional
            The name of the sheet in the Excel file containing the triangle data. If not provided, the first sheet will be used.
        sheet_range : str, optional
            A string containing the range of cells to read from the Excel file. The range should be in the format "A1:B2".
        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the Excel file.
        """
        # Read data from the Excel file
        if sheet_range:
            # If a range is provided, read only the specified range
            _, idx = range_to_tuple(f"'{sheet_name}'!{sheet_range}")
            row1, col1 = idx[0]-1, idx[1]-1
            row2, col2 = idx[2]-1, idx[3]

            # read in the subset of the excel file
            df = pd.read_excel(filename, header=0, sheet_name=sheet_name).iloc[row1:row2, col1:col2]

            # set the column names as the first row
            # df.columns = df.iloc[0]

            # # drop the first row
            # df.drop(df.index[0], inplace=True)
        else:
            # If no range is provided, read the entire sheet
            df = pd.read_excel(filename, header=0, sheet_name=sheet_name)

        # Set the origin period as the index
        df.set_index(df.columns.tolist()[:origin_columns], inplace=True)

        # If the columns are numeric, convert them to integer categories
        df.columns = df.columns.astype(int)

        # re-sort the columns
        df.sort_index(axis=1, inplace=True)

        # If there are rows with all zeros, or all NaNs, drop them
        df.dropna(axis=0, how='all', inplace=True)

        # If there are any remaining columns with all zeros, or all NaNs, drop them
        df.dropna(axis=1, how='all', inplace=True)

        # Create and return a Triangle object
        return cls(id=id, tri=df, triangle=df)
    
    @classmethod
    def from_taylor_ashe(cls) -> 'Triangle':
        """
        Create a Triangle object from the Taylor Ashe sample data.
        Parameters:
        -----------
        None
        Returns:
        --------
        Triangle
            A Triangle object with data loaded from the Taylor Ashe sample data.
        """
        # Get the current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        # Construct the file path to the sample data
        data_file = os.path.join(current_dir, 'data', 'taylorashe.csv')
        
        # Read the data from the CSV file
        df = pd.read_csv(data_file, header=0, index_col=0)

        # Create and return a Triangle object
        return cls(id='paid_loss', tri=df, triangle=df)
    
    @classmethod
    def from_dahms(cls) -> tuple:
        """
        Create a Triangle object from the Dahms sample data. This sample data contains
        both a reported and a paid triangle, so this method returns a tuple containing
        both triangles.

        Return is of the form (rpt, paid).

        Parameters:
        -----------
        None

        Returns:
        --------
        tuple[Triangle, Triangle]
            A tuple containing a Triangle object with data loaded from the reported
            triangle, and a Triangle object with data loaded from the paid triangle.
        """
        # Get the current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        # Construct the file path to the sample data
        data_file = os.path.join(current_dir, 'papers', 'dahms reserve triangles.xlsx')
        
        # Read the data from the CSV file
        paid = cls.from_excel(data_file, sheet_name='paid', id="paid_loss", origin_columns=1, sheet_range="a1:k11")
        rpt = cls.from_excel(data_file, sheet_name='rpt', id="rpt_loss", origin_columns=1, sheet_range="a1:k11")

        # Create and return a Triangle object
        return rpt, paid
    
    # return a data frame formatted with a background color gradient for the columns
    def col_gradient(self,
                     cmap: str = 'RdYlGn',
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                        ) -> pd.DataFrame:
        """
        Return a data frame formatted with a background color gradient for the columns.
        Parameters:
        -----------
        cmap : str, default='RdYlGn'
            The name of the matplotlib colormap to use. Popular options include:
            1. 'RdYlGn'
            2. 'RdBu'
            3. 'PuOr'
            4. 'RdYlBu'
            5. 'RdGy'
            6. 'RdPu'
        vmin : float, optional
            The minimum value to use for the colormap. If not provided, the minimum
            value in the triangle will be used.
        vmax : float, optional
            The maximum value to use for the colormap. If not provided, the maximum
            value in the triangle will be used.
        Returns:
        --------
        pd.DataFrame
            A data frame formatted with a background color gradient for the columns.
        """ 
        # Get the minimum and maximum values
        # if vmin is None:
        #     vmin = self.tri.min().min()
        # if vmax is None:
        #     vmax = self.tri.max().max()

        # Get the colormap
        cmap = plt.get_cmap(cmap)

        # Get the normalized values
        # norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # # Get the colors
        # colors = [cmap(norm(value)) for value in self.tri.values.flatten()]

        # # Get the number of columns
        # n_cols = self.tri.shape[1]

        # # Get the number of rows
        # n_rows = self.tri.shape[0]

        # # Reshape the colors
        # colors = np.array(colors).reshape(n_rows, n_cols, 4)

        # # create a data frame with the color gradient
        # df = pd.DataFrame(colors[:2], index=self.tri.index, columns=self.tri.columns)

        # Return the data frame
        return self.tri.style.background_gradient(cmap=cmap, vmin=vmin, vmax=vmax)



    ## Data loading methods
    def from_dataframe(self,
                       df: pd.DataFrame,
                       use_index: bool = True,
                       use_columns: bool = True,
                       origin_col: str = 'ay',
                       development_col: str = 'dev_month',
                       loss_col: str = 'loss',
                       return_df: bool = False
                       ) -> None:
        """
        Create a `Triangle` object from a pandas dataframe.
        Parameters:
        -----------
        df: `pd.DataFrame`
            The triangle data.
        use_index: `bool`, default=`True`
            If `True`, use the dataframe index as the origin period.
        use_columns: `bool`, default=`True`
            If `True`, use the dataframe columns as the development period.
        origin_col: `str`, default=`'ay'`
            The column name to use for the origin period. Only used
            if `use_index` is `False`.
        development_col: `str`, default=`'dev_month'`
            The column name to use for the development period. Only
            used if `use_columns` is `False`.
        return_df: `bool`, default=`False`
            If `True`, return the triangle data as a pandas dataframe.
        Returns:
        --------
        `None`
        """
        # if use_index is True, use the dataframe index as the origin period
        if use_index:
            origin_col = df.index.name

        # if use_columns is True, use the dataframe columns as the development period
        if use_columns:
            development_col = df.columns.name

        # pivot the dataframe to get the triangle data
        self.triangle.tri = df.pivot_table(index=origin_col, columns=development_col, values=loss_col)

    def cum_to_inc(self,
                   cum_tri : pd.DataFrame = None,
                   _return: bool = False
                   ) -> pd.DataFrame:
        """
        Convert cumulative triangle data to incremental triangle data.

        Parameters:
        -----------
        cum_tri: pd.DataFrame
            The cumulative triangle data. Default is None, in which case
            the triangle data from the Triangle object is used.
        _return: bool
            If True, return the incremental triangle data. Default is False.

        Returns:
        --------
        inc_tri: pd.DataFrame
            The incremental triangle data.
        """
        # get the cumulative triangle data
        if cum_tri is None:
            cum_tri = self.triangle

        # get the cumulative triangle data
        inc_tri = cum_tri - cum_tri.shift(1, axis=1, fill_value=0)

        # set the incremental triangle data
        self.incr_triangle = inc_tri

        # return the incremental triangle data
        if _return:
            return inc_tri
        
    ## Basic triangle methods
    def _ata_tri(self) -> None:
        """
        Calculate the age-to-age factor triangle from the triangle data.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # instantiate the ata triangle (same shape as the triangle data)
        ata = pd.DataFrame(np.zeros(self.tri.shape),
                           index=self.tri.index,
                           columns=self.tri.columns)
        
        # if there are any values of 0 in the triangle data, set them to nan
        self.tri[self.tri == 0] = np.nan

        # loop through the columns in the triangle (excl. the last column)
        for i in range(self.tri.shape[1] - 1):
            # calculate the age-to-age factor
            ata.iloc[:, i] = self.tri.iloc[:, i + 1] / self.tri.iloc[:, i]

        # set the last column of the ata triangle to nan
        ata.iloc[:, self.tri.shape[1] - 1] = np.nan

        return ata
    
    def _vwa(self, n: int = None, tail: float = 1.0) -> pd.DataFrame:
        """
        Calculate the volume weighted average (VWA) of the triangle data.

        Parameters:
        -----------
        n: int
            The number of periods to use in the VWA calculation. If None, use 
            all available periods.
        tail: float
            The tail factor to use in the VWA calculation. Default is 1.0, or 
            no tail factor.

        Returns:
        --------
        vwa: pandas dataframe
            The VWA triangle data.
        """
        # instantiate the vwa results - a series whose length is equal to the number of
        # columns in the triangle data, with the index set to the column names
        vwa = pd.Series(np.zeros(self.tri.shape[1]), index=self.tri.columns, dtype=float)

        # if n is None, use all available periods
        is_all = n is None

        # need a value for n in the loop below
        n2 = n if n is not None else self.tri.shape[0]

        # loop through the columns in the triangle data (excl. the last column)
        for i in range(self.tri.shape[1] - 1):
            next_col = self.tri.iloc[:, i + 1]
            cur_col = self.tri.iloc[:, i]
            if is_all or next_col.dropna().shape[0] <= n2:
                num = next_col.sum()
                den = cur_col.mask(next_col.isna(), np.nan).sum()
            else:
                num = next_col.dropna().tail(n).sum()
                den = cur_col.mask(next_col.isna(), np.nan).dropna().tail(n).sum()

            vwa.iloc[i] = num / den

        # set the last column of the vwa results to the tail factor
        vwa.iloc[self.tri.shape[1] - 1] = tail

        return vwa

    
    def _ave_ata(self,
                 n: int = None,
                 tail: float = 1.0
                 ) -> pd.Series:
        """
        Calculate the average age-to-age factor (Ave-ATA) of the triangle data.

        Parameters:
        -----------
        n: int
            The number of periods to use in the Ave-ATA calculation. If None, use 
            all available periods.
        tail: float
            The tail factor to use in the Ave-ATA calculation. Default is 1.0, or 
            no tail factor.

        Returns:
        --------
        ave_ata: pd.Series
            The Ave-ATA triangle data. Shape is the same as the number of columns
            in the triangle data, with the index set to the column names.
        """
        # instantiate the ave-ata results - a series whose length is equal to the number of
        # columns in the triangle data, with the index set to the column names
        ave_ata = pd.Series(np.zeros(self.tri.shape[1]), index=self.tri.columns, dtype=float)

        # if n is None, use all available periods
        is_all = n is None

        # need a value for n in the loop below
        n2 = n if n is not None else self.tri.shape[0]

        # # get the triangle
        # tri = self.tri

        # # get the triangle of age-to-age factors
        # ata = self._ata_tri()

        # loop through the columns in the triangle data (excl. the last column)
        for i, column in enumerate(self.tri.columns[:-1]):
            # calculate the Ave-ATA -- if n is None, use all available periods
            # otherwise, use the last n periods, until the number of periods
            # is less than n (in which case, use all available periods)
            if is_all or self.tri.iloc[:, i + 1].dropna().shape[0] <= n2:
                ave_ata[column] = self._ata_tri().iloc[:, i].mean(skipna=True)
            else:
                # drop the na values so they aren't included in the average,
                # then average the previous n periods
                ave_ata[column] = self._ata_tri().iloc[:, i].dropna().tail(n).mean(skipna=True)

        # set the last column of the ave-ata results to the tail factor
        ave_ata[self.tri.columns[-1]] = tail

        return ave_ata

    
    def _medial_ata(self,
                    n: int = None,
                    tail: float = 1.0,
                    excludes: str = 'hl'
                    ) -> pd.Series:
        """
        Calculate the medial age-to-age factor (Medial-ATA) of the triangle data. This
        excludes one or more of the values in the average calculation. Once the values are
        removed, the average is calculated as a normal average. 

        Parameters:
        -----------
        n: int
            The number of periods to use in the Medial-ATA calculation. If None, use
            all available periods.
        tail: float
            The tail factor to use in the Medial-ATA calculation. Default is 1.0, or
            no tail factor.
        excludes: str
            The exclusions to use in the average calculation. Default is 'hl', 
            or high and low. If ave_type is 'triangle', this parameter is ignored.
            This parameter is a string of characters, where each character is an
            exclusion. The options are:
                h - high
                l - low
                m - median
            These characters can be in any order, and any number of them can be
            specified. For example, 'hl' excludes the high and low values, as does
            'lh', but 'hhl' excludes only the high value.

        Returns:
        --------
        medial_ata: pd.Series
            The Medial-ATA triangle data. Shape is the same as the number of columns
            in the triangle data, with the index set to the column names.
        """
        # instantiate the medial-ata results - a series whose length is equal to the number of
        # columns in the triangle data, with the index set to the column names
        medial_ata = pd.Series(np.zeros(self.tri.shape[1]), index=self.tri.columns, dtype=float)

        # default if can't calculate this is to use the simple average
        default = self._vwa(n=n, tail=tail)

        # if n is None, use all available periods
        is_all = n is None

        # if the string contains 'h', exclude the high value, 'l' excludes the low value,
        # and 'm' excludes the median value
        exclude_high = 'h' in excludes.lower()
        exclude_low = 'l' in excludes.lower()
        exclude_median = 'm' in excludes.lower()

        # need a value for n in the loop below
        n2 = n if n is not None else self.tri.shape[0]

        # loop through the columns in the triangle data (excl. the last column)
        for i, column in enumerate(self.tri.columns[:-1]):
            # temp column:
            temp_column = (self._ata_tri()).iloc[:, i].dropna()

            # count that there are enough values to calculate the average
            need_at_least = exclude_high + exclude_low + exclude_median

            # if there are not enough values to calculate the average, use the default
            if temp_column.shape[0] <= need_at_least:
                medial_ata[column] = default[column]
                continue
            else:



                # if we are not using all available periods, filter so only have 
                # the last n periods available
                if is_all or self.tri.iloc[:, i + 1].dropna().shape[0] <= n2:
                    temp_column = temp_column.dropna()
                else:
                    temp_column = temp_column.dropna().tail(n)

                

                # if we are excluding the high value, remove it (same with low and median)
                if exclude_high:
                    temp_column = temp_column.drop(temp_column.idxmax())
                if exclude_low:
                    temp_column = temp_column.drop(temp_column.idxmin())
                if exclude_median:
                    temp_column = temp_column.drop(temp_column.median())

                ## calculate the Medial-ATA
                medial_ata[column] = temp_column.mean(skipna=True)

        # set the last column of the medial-ata results to the tail factor
        medial_ata[self.tri.columns[-1]] = tail

        return medial_ata


    
    def ata(self,
            ave_type: str = 'triangle',
            n: int = None,
            tail: float = 1.0,
            excludes: str = 'hl'
            ) -> pd.DataFrame:
        """
        Returns the age-to-age factors of the triangle data, depending on the
        average type. Default is the triangle of age-to-age factors, but passing 
        'vwa', 'simple', or 'medial' will return the volume weighted average,
        simple average, or medial average age-to-age factors, respectively. If one 
        of the averages is selected, the number of periods to use in the average,
        tail factor, and exclusions can be specified (or they will use the defaults).

        Parameters:
        -----------
        ave_type: str
            The type of average to use. Options are 'triangle', 'vwa', 'simple', 
            and 'medial'. Default is 'triangle'.
        n: int
            The number of periods to use in the average calculation. If None, use 
            all available periods. If ave_type is 'triangle', this parameter is
            ignored.
        tail: float
            The tail factor to use in the average calculation. Default is 1.0, or 
            no tail factor. If ave_type is 'triangle', this parameter is ignored.
        excludes: str
            The exclusions to use in the average calculation. Default is 'hl', 
            or high and low. If ave_type is 'triangle', this parameter is ignored.
            This parameter is a string of characters, where each character is an
            exclusion. The options are:
                h - high
                l - low
                m - median
            These characters can be in any order, and any number of them can be
            specified. For example, 'hl' excludes the high and low values, as does
            'lh', but 'hhl' excludes only the high value.
                
            
        Returns:
        --------
        ata: pd.DataFrame
            The age-to-age factors of the triangle data, depending on the average
            type. Shape is the same as the triangle data.
        """
        # if the average type is 'triangle', return the triangle of age-to-age factors
        if ave_type.lower() == 'triangle':
            return self._ata_tri()
        # if the average type is 'vwa', return the volume weighted average age-to-age factors
        elif ave_type.lower() == 'vwa':
            return self._vwa(n=n, tail=tail)
        # if the average type is 'simple', return the simple average age-to-age factors
        elif ave_type.lower() == 'simple':
            return self._ave_ata(n=n, tail=tail)
        # if the average type is 'medial', return the medial average age-to-age factors
        elif ave_type.lower() == 'medial':
            return self._medial_ata(n=n, tail=tail, excludes=excludes)
        # if the average type is not recognized, raise an error
        else:
            raise ValueError('Invalid age-to-age type. Must be "triangle", "vwa", "simple", or "medial"')
        
    def atu(self,
            ave_type: str = 'vwa',
            n: int = None,
            tail: float = 1.0,
            excludes: str = 'hl',
            custom: np.ndarray = None
            ) -> pd.DataFrame:
        """
        Calculates the age-to-ultimate factors from the triangle data.

        Parameters:
        -----------
        ave_type: str
            The type of average to use. Options are 'vwa', 'simple', 
            and 'medial'. Default is 'vwa'.
        n: int
            The number of periods to use in the average calculation. If None, use 
            all available periods. 
        tail: float
            The tail factor to use in the average calculation. Default is 1.0, or 
            no tail factor.
        excludes: str
            The exclusions to use in the average calculation. Default is 'hl', 
            or high and low. This parameter is a string of characters, where each 
            character is an exclusion. The options are:
                h - high
                l - low
                m - median
            These characters can be in any order, and any number of them can be
            specified. For example, 'hl' excludes the high and low values, as does
            'lh', but 'hhl' excludes only the high value.
        custom: np.ndarray
            A custom array of age-to-age factors to use in the calculation. If
            None, use the age-to-age factors calculated from the 'ave_type'.
            If not None, the 'ave_type', 'n', 'tail', and 'excludes' parameters
            are ignored.
            Default is None.

        Returns:
        --------
        atu: pd.DataFrame
            The age-to-ultimate factors of the triangle data.
        """
        # calculate the age-to-age factors
        if custom is None:
            age_to_age = self.ata(ave_type=ave_type, n=n, tail=tail, excludes=excludes)
        else:
            age_to_age = pd.Series(custom, index=self.tri.columns)

        # calculate the age-to-ultimate factors (cumulative product of the ata factors, 
        # starting with the last column/the tail factor)
        age_to_ult = age_to_age[::-1].cumprod()[::-1]

        return age_to_ult
    
    def diag(self, calendar_year: int = None) -> pd.DataFrame:
        """
        Calculates the specified diagonal of the triangle data.

        Parameters:
        -----------
        calendar_year: int
            The calendar year of the diagonal to return. If None, return the
            current diagonal. Default is None.
            This is not implemented.

        Returns:
        --------
        diag: pd.DataFrame
            The diagonal of the triangle data.
        """
        # look at the triangle as an array
        triangle_array = self.tri.to_numpy()



        # if the calendar year is not specified, return the current diagonal
        if calendar_year is None:
            calendar_year = triangle_array.shape[0]

        # diagonal is a series of length equal to the number of rows in the triangle
        diag = pd.Series(np.diagonal(np.fliplr(triangle_array)), index=self.tri.index)

        # match the index of the diagonal to column name that value can be found in
        # (remember that the triangle may not be the same size as the index, if the
        # triangle is not square -- so we need to actually match the first occurrence
        # of the value to the column name)
################################################################################################################################################################




        return diag
    
    def ata_summary(self) -> pd.DataFrame:
        """
        Produces a fixed summary of the age-to-age factors for the triangle data.

        Contains the following:
            - Triangle of age-to-age factors
            - Volume weighted average age-to-age factors for all years, 5 years, 3 years, and 2 years
            - Simple average age-to-age factors for all years, 5 years, 3 years, and 2 years
            - Medial average age-to-age factors for 5 years, excluding high, low, and high/low values
        """

        triangle = self

        ata_tri = triangle.ata().round(3)

        vol_wtd  = pd.DataFrame({
            'Vol Wtd': pd.Series(["" for _ in range(ata_tri.shape[1]+1)], index=ata_tri.reset_index().columns)
            , 'All Years': triangle.ata('vwa').round(3)
            , '5 Years': triangle.ata('vwa', 5).round(3)
            , '3 Years': triangle.ata('vwa', 3).round(3)
            , '2 Years': triangle.ata('vwa', 2).round(3)
            }).transpose()

        simple = pd.DataFrame({
            'Simple': pd.Series(["" for _ in range(ata_tri.shape[1]+1)], index=ata_tri.reset_index().columns),
            'All Years': triangle.ata('simple').round(3),
            '5 Years': triangle.ata('simple', 5).round(3),
            '3 Years': triangle.ata('simple', 3).round(3),
            '2 Years': triangle.ata('simple', 2).round(3)}).transpose()

        medial = pd.DataFrame({
            'Medial 5-Year': pd.Series(["" for _ in range(ata_tri.shape[1]+1)], index=ata_tri.reset_index().columns),
            'Ex. Hi/Low': triangle.ata('medial', 5, excludes='hl').round(3),
            'Ex. Hi': triangle.ata('medial', 5, excludes='h').round(3),
            'Ex. Low': triangle.ata('medial', 5, excludes='l').round(3),
                                }).transpose()

        out = (pd.concat([ata_tri.drop(index=10), vol_wtd, simple, medial], axis=0)
               .drop(columns=self.tri.columns[-1])
               .fillna(''))
        
        # check to see if the last column is all '' (empty strings)
        if out.iloc[:, -1].str.contains('').all():
            out = out.drop(columns=out.columns[-1])
        
        return out

    def melt_triangle(self,
                      id_cols: list = None,
                      var_name: str = 'dev',
                      value_name: str = None,
                      _return: bool = True,
                      incr_tri: bool = True
                      ) -> pd.DataFrame:
        """
        Melt the triangle data into a single column of values.
        Parameters:
        -----------
        id_cols: list
            The columns to use as the id variables. Default is None, in which
            case the index is used.
        var_name: str
            The name of the variable column. Default is 'dev'.
        value_name: str
            The name of the value column. Default is None, in which case
            the value column is set equal to the triangle ID.
        _return: bool
            If True, return the melted triangle data as a pandas dataframe.
            Default is True.
        incr_tri: bool
            If True, use the incremental triangle data. Default is True. If
            False, use the cumulative triangle data.

        Returns:
        --------
        melted: pd.DataFrame
            The melted triangle data.
        """
        # if id_cols is None, use the index
        if id_cols is None:
            id_cols = self.triangle.index.name

        # if value_name is None, use the triangle ID
        if value_name is None:
            value_name = self.id

        # get the triangle data
        if incr_tri:
            if self.incr_triangle is None:
                self.cum_to_inc()
            tri = self.incr_triangle
        else:
            tri = self.triangle

        # melt the triangle data
        melted = (tri
                    .reset_index()
                    .melt(id_vars=id_cols,
                          var_name=var_name,
                          value_name=value_name)
                 )
        
        # if _return is True, return the melted triangle data
        if _return:
            return melted
        

    # def melt(self
    #          , value_col : str = None
    #          , origin_col : str = None
    #          , dev_col : str = None
    #          ) -> pd.DataFrame:
    #     """
    #     This is a convenience function to use when the triangle data is formatted
    #     properly, meaning the index is the origin period and the columns are the
    #     development periods.

    #     Parameters:
    #     -----------
    #     value_col: str
    #         The name of the value column. Default is None, in which case the
    #         value column is set equal to the triangle ID.
    #     origin_col: str
    #         The name of the origin column. Default is None, in which case the
    #         origin column is set equal to the index name.
    #     dev_col: str
    #         The name of the development column. Default is None, in which case
    #         the development column is set equal to the column names.

    #     Returns:
    #     --------
    #     melted: pd.DataFrame
    #         The melted triangle data.
    #     """
    #     # if value_col is None, use the triangle ID
    #     if value_col is None:
    #         value_col = self.id

    #     # if origin_col is None, use the index name
    #     if origin_col is None:
    #         origin_col = self.triangle.index.name

    #     # if dev_col is None, use the column names
    #     if dev_col is None:
    #         dev_col = self.triangle.columns

    #     # melt the triangle data
    #     melted = self.melt_triangle(self,
    #                   id_cols = origin_col,
    #                   var_name = dev_col,
    #                   value_name = value_col,
    #                   _return = True,
    #                   incr_tri = False
    #                   )
        
    #     return melted
        

        
        
    def base_design_matrix(self,
                           id_cols: list = None,
                           var_name: str = 'dev',
                           value_name: str = None,
                           trends: bool = True,
                           _return: bool = True,
                           incr_tri: bool = True
                           ) -> pd.DataFrame:
        """
        Creates a design matrix from the triangle data. The design matrix is a pandas
        dataframe with one row for each triangle cell, and one column for each origin
        and development period. The origin and development periods are encoded as
        dummy variables, and if `trends` is True, the origin and development periods
        are also encoded as linear trends, instead of just dummy variables.

        This is the base design matrix for a rocky3 model. The base design matrix
        is used to create the full design matrix, which includes any interaction
        terms, and any other covariates.

        All diagnostics will implicitly check that any changes to the base model provide
        improvements to the base model fit.

        Parameters:
        -----------
        id_cols: list
            The columns to use as the id variables. Default is None, in which
            case the index is used.
        var_name: str
            The name of the variable column. Default is 'dev'.
        value_name: str
            The name of the value column. Default is None, in which case
            the value column is set equal to the triangle ID.
        trends: bool
            If True, include linear trends in the design matrix. Default is True.
        _return: bool
            If True, return the design matrix as a pandas dataframe.
            Default is True.
        incr_tri: bool
            If True, use the incremental triangle data. Default is True. If
            False, use the cumulative triangle data.
            
        Returns:
        --------
        dm_total: pd.DataFrame
            The design matrix.
        """
        if id_cols is None:
            id_cols = self.triangle.index.name

        if value_name is None:
            value_name = self.id

        # melt the triangle data
        melted = self.melt_triangle(id_cols=id_cols,
                                             var_name=var_name,
                                             value_name=value_name,
                                             _return=True,
                                             incr_tri=incr_tri)
        
        # convert the origin and development periods to zero-padded categorical variables
        melted['AY'] = melted['AY'].astype(str).str.zfill(4).astype('category')
        melted['dev'] = melted['dev'].astype(str).str.zfill(4).astype('category')

        # create the design matrix
        dm_total = pd.get_dummies(melted, columns=['AY', 'dev'], drop_first=True)

        # if trends is True, add linear trends to the design matrix
        if trends:
            # create dummy variables for the origin and development periods
            dm_ay = pd.get_dummies(melted[['AY']], drop_first=True)
            dm_dev = pd.get_dummies(melted[['dev']], drop_first=True)

            # ay dm columns
            cols = dm_ay.columns.tolist()
            
            # reverse the order of the columns (to loop backwards)
            cols.reverse()

            # loop backwards through the columns
            for i, c in enumerate(cols):
                # if i==0, set the column equal to itself
                if i==0:
                    dm_total[c] = dm_ay[c]
                
                # otherwise, add the column to the previous column
                else:
                    dm_total[c] = dm_ay[c] + dm_total[cols[i-1]]

            # do the same thing for the development period dummy variables
            cols = dm_dev.columns.tolist()
            cols.reverse()
            for i, c in enumerate(cols):
                if i==0:
                    dm_total[c] = dm_dev[c]
                else:
                    dm_total[c] = dm_dev[c] + dm_total[cols[i-1]]

            # add a column called "is_observed" at the beginning that is 1 if
            # dm_total[value_name] is not null and 0 otherwise
            observed_col = dm_total[value_name].notnull().astype(int)
            dm_total.insert(0, 'is_observed', observed_col)

        # if _return is True, return the design matrix
        if _return:
            return dm_total
        
        
        self.X_base = dm_total.drop(columns=value_name)
        self.y_base = dm_total[value_name]
        
        # ay/dev id for each row
        self.X_id = pd.DataFrame(dict(ay=melted['AY'].astype(int).values, dev=melted['dev'].astype(int).values))
        self.X_id['cal'] = self.X_id.ay + self.X_id.dev - 1
        self.X_id.index = self.X_base.index

    def base_linear_model(self,
                          id_cols: list = None,
                          var_name: str = 'dev',
                          value_name: str = None,
                          trends: bool = True,
                          incr_tri: bool = True,
                          intercept_ : bool = True
                          ) -> pd.DataFrame:
        """
        Builds the train/forecast data split based off of the base design matrix.

        Parameters:
        -----------
        (See base_design_matrix() for parameter descriptions of id_cols, var_name,
        value_name, trends, _return, and incr_tri)

        intercept_: bool
            If True, include an intercept in the model. Default is True.

        Returns:
        --------
        dm_base_train: pd.DataFrame
            The training data design matrix with target variable as the first column.
        """
        # if the base design matrix has not been created, create it
        if self.X_base is None:
            self.base_design_matrix(id_cols=id_cols,
                                    var_name=var_name,
                                    value_name=value_name,
                                    trends=trends,
                                    _return=False,
                                    incr_tri=incr_tri)
            
        # create the train/forecast data split based on whether or not the
        # target variable is null
        self.X_base_train = self.X_base[self.X_base.is_observed.eq(1)].rename(columns={'is_observed': 'intercept'})
        self.y_base_train = self.y_base[self.X_base.is_observed.eq(1)]
        self.X_base_forecast = self.X_base[self.X_base.is_observed.eq(0)].rename(columns={'is_observed': 'intercept'}).assign(intercept=1)

        self.X_id_train = self.X_id[self.X_base.is_observed.eq(1)]
        self.X_id_forecast = self.X_id[self.X_base.is_observed.eq(0)]

        # if intercept_ is False, drop the intercept column
        if not intercept_:
            self.X_base_train = self.X_base_train.drop(columns='intercept')
            self.X_base_forecast = self.X_base_forecast.drop(columns='intercept')
            #

