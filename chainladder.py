# """
# This module contains the Deterministic class, which is a subclass of the
# BaseEstimator class. The Deterministic class is used to fit tradiational actuarial
# models to data. The Deterministic class is used to fit the following models:
# - Mack chain ladder
# - Bornhuetter-Ferguson
# - Cape Cod
# - Benktander
# - Hurlimann
# - Brossius
# """

# from BaseEstimator import BaseEstimator
# # from triangle import Triangle
# from utils import formatter

# import numpy as np
# import pandas as pd

# from dataclasses import dataclass

# @dataclass
# class ChainLadder(BaseEstimator):
#     # def __init__(self,
#     #              name: str = None,
#     #              triangle: str = None
#     #              ) -> None:
#     """
#     Deterministic actuarial methods base class.

#     Parameters:
#     -----------
#     name: `str`, default=`None`
#         The name of the estimator. This is required, and is used to distinguish
#         between different models attached to the same ROCKY3 object.
#         Raises an error if name is None.
#     triangle: `str`, default=`None`
#         The name of the triangle to be used for the model. This is required, and
#         is used to distinguish between different triangles attached to the same
#         ROCKY3 object. Raises an error if triangle is None.
    
#     """
#         # super().__init__(name=name)
#     name: str = None
#     triangle: str = None

#     ## Basic triangle methods
#     def _ata_tri(self) -> None:
#         """
#         Calculate the age-to-age factor triangle from the triangle data.

#         Parameters:
#         -----------
#         None

#         Returns:
#         --------
#         None
#         """
#         # instantiate the ata triangle (same shape as the triangle data)
#         ata = pd.DataFrame(np.zeros(self.tri.shape),
#                            index=self.tri.index,
#                            columns=self.tri.columns)
        
#         # if there are any values of 0 in the triangle data, set them to nan
#         self.tri[self.tri == 0] = np.nan

#         # loop through the columns in the triangle (excl. the last column)
#         for i in range(self.tri.shape[1] - 1):
#             # calculate the age-to-age factor
#             ata.iloc[:, i] = self.tri.iloc[:, i + 1] / self.tri.iloc[:, i]

#         # set the last column of the ata triangle to nan
#         ata.iloc[:, self.tri.shape[1] - 1] = np.nan

#         return ata
    
#     def _vwa(self, n: int = None, tail: float = 1.0) -> pd.DataFrame:
#         """
#         Calculate the volume weighted average (VWA) of the triangle data.

#         Parameters:
#         -----------
#         n: int
#             The number of periods to use in the VWA calculation. If None, use 
#             all available periods.
#         tail: float
#             The tail factor to use in the VWA calculation. Default is 1.0, or 
#             no tail factor.

#         Returns:
#         --------
#         vwa: pandas dataframe
#             The VWA triangle data.
#         """
#         # instantiate the vwa results - a series whose length is equal to the number of
#         # columns in the triangle data, with the index set to the column names
#         vwa = pd.Series(np.zeros(self.tri.shape[1]), index=self.tri.columns)

#         # if n is None, use all available periods
#         is_all = n is None

#         # need a value for n in the loop below
#         n2 = n if n is not None else self.tri.shape[0]

#         # loop through the columns in the triangle data (excl. the last column)
#         for i in range(self.tri.shape[1] - 1):
#             next_col = self.tri.iloc[:, i + 1]
#             cur_col = self.tri.iloc[:, i]
#             # calculate the VWA -- if n is None, use all available periods
#             # otherwise, use the last n periods, until the number of periods
#             # is less than n (in which case, use all available periods)
#             if is_all | next_col.dropna().shape[0] <= n2:
#                 num = next_col.sum(skipna=True)
#                 den = cur_col.mask(next_col.isna(), np.nan).sum(skipna=True)
#             else:
#                 # drop the na values so they aren't included in the sum, then sum the previous n periods
#                 num = next_col.dropna().tail(n).sum(skipna=True)
#                 den = cur_col.mask(next_col.isna(), np.nan).dropna().tail(n ).sum(skipna=True)

#             vwa[i] = num / den

#         # set the last column of the vwa results to the tail factor
#         vwa.iloc[self.tri.shape[1] - 1] = tail

#         return vwa
    
#     def _ave_ata(self,
#                  n: int = None,
#                  tail: float = 1.0
#                  ) -> pd.Series:
#         """
#         Calculate the average age-to-age factor (Ave-ATA) of the triangle data.

#         Parameters:
#         -----------
#         n: int
#             The number of periods to use in the Ave-ATA calculation. If None, use 
#             all available periods.
#         tail: float
#             The tail factor to use in the Ave-ATA calculation. Default is 1.0, or 
#             no tail factor.

#         Returns:
#         --------
#         ave_ata: pd.Series
#             The Ave-ATA triangle data. Shape is the same as the number of columns
#             in the triangle data, with the index set to the column names.
#         """
#         # instantiate the ave-ata results - a series whose length is equal to the number of
#         # columns in the triangle data, with the index set to the column names
#         ave_ata = pd.Series(np.zeros(self.tri.shape[1]), index=self.tri.columns)

#         # if n is None, use all available periods
#         is_all = n is None

#         # need a value for n in the loop below
#         n2 = n if n is not None else self.tri.shape[0]

#         # # get the triangle
#         # tri = self.tri

#         # # get the triangle of age-to-age factors
#         # ata = self._ata_tri()

#         # loop through the columns in the triangle data (excl. the last column)
#         for i in range(self.tri.shape[1] - 1):
#             # calculate the Ave-ATA -- if n is None, use all available periods
#             # otherwise, use the last n periods, until the number of periods
#             # is less than n (in which case, use all available periods)
#             if is_all | self.tri.iloc[:, i + 1].dropna().shape[0] <= n2:
#                 ave_ata[i] = self._ata_tri().iloc[:, i].mean(skipna=True)
#             else:
#                 # drop the na values so they aren't included in the average,
#                 # then average the previous n periods
#                 ave_ata[i] = self._ata_tri().iloc[:, i].dropna().tail(n).mean(skipna=True)
                
#         # set the last column of the ave-ata results to the tail factor
#         ave_ata.iloc[self.tri.shape[1] - 1] = tail

#         return ave_ata
    
#     def _medial_ata(self,
#                     n: int = None,
#                     tail: float = 1.0,
#                     excludes: str = 'hl'
#                     ) -> pd.Series:
#         """
#         Calculate the medial age-to-age factor (Medial-ATA) of the triangle data. This
#         excludes one or more of the values in the average calculation. Once the values are
#         removed, the average is calculated as a normal average. 

#         Parameters:
#         -----------
#         n: int
#             The number of periods to use in the Medial-ATA calculation. If None, use
#             all available periods.
#         tail: float
#             The tail factor to use in the Medial-ATA calculation. Default is 1.0, or
#             no tail factor.
#         excludes: str
#             The exclusions to use in the average calculation. Default is 'hl', 
#             or high and low. If ave_type is 'triangle', this parameter is ignored.
#             This parameter is a string of characters, where each character is an
#             exclusion. The options are:
#                 h - high
#                 l - low
#                 m - median
#             These characters can be in any order, and any number of them can be
#             specified. For example, 'hl' excludes the high and low values, as does
#             'lh', but 'hhl' excludes only the high value.

#         Returns:
#         --------
#         medial_ata: pd.Series
#             The Medial-ATA triangle data. Shape is the same as the number of columns
#             in the triangle data, with the index set to the column names.
#         """
#         # instantiate the medial-ata results - a series whose length is equal to the number of
#         # columns in the triangle data, with the index set to the column names
#         medial_ata = pd.Series(np.zeros(self.tri.shape[1]), index=self.tri.columns)

#         # default if can't calculate this is to use the simple average
#         default = self._vwa(n=n, tail=tail)

#         # if n is None, use all available periods
#         is_all = n is None

#         # if the string contains 'h', exclude the high value, 'l' excludes the low value,
#         # and 'm' excludes the median value
#         exclude_high = 'h' in excludes.lower()
#         exclude_low = 'l' in excludes.lower()
#         exclude_median = 'm' in excludes.lower()

#         # need a value for n in the loop below
#         n2 = n if n is not None else self.tri.shape[0]

#         # loop through the columns in the triangle data (excl. the last column)
#         for i in range(self.tri.shape[1] - 1):
#             # temp column:
#             temp_column = (self._ata_tri()).iloc[:, i].dropna()

#             # count that there are enough values to calculate the average
#             need_at_least = exclude_high + exclude_low + exclude_median

#             # if there are not enough values to calculate the average, use the default
#             if temp_column.shape[0] <= need_at_least:
#                 medial_ata[i] = default[i]
#                 continue
#             else:


#                 # if we are not using all available periods, filter so only have 
#                 # the last n periods available
#                 if is_all | self.tri.iloc[:, i + 1].dropna().shape[0] <= n2:
#                     temp_column = temp_column.dropna()
#                 else:
#                     temp_column = temp_column.dropna().tail(n)

                

#                 # if we are excluding the high value, remove it (same with low and median)
#                 if exclude_high:
#                     temp_column = temp_column.drop(temp_column.idxmax())
#                 if exclude_low:
#                     temp_column = temp_column.drop(temp_column.idxmin())
#                 if exclude_median:
#                     temp_column = temp_column.drop(temp_column.median())

#                 # calculate the Medial-ATA
#                 medial_ata[i] = temp_column.mean(skipna=True)

#         # set the last column of the medial-ata results to the tail factor
#         medial_ata.iloc[self.tri.shape[1] - 1] = tail

#         return medial_ata

    
#     def ata(self,
#             ave_type: str = 'triangle',
#             n: int = None,
#             tail: float = 1.0,
#             excludes: str = 'hl'
#             ) -> pd.DataFrame:
#         """
#         Returns the age-to-age factors of the triangle data, depending on the
#         average type. Default is the triangle of age-to-age factors, but passing 
#         'vwa', 'simple', or 'medial' will return the volume weighted average,
#         simple average, or medial average age-to-age factors, respectively. If one 
#         of the averages is selected, the number of periods to use in the average,
#         tail factor, and exclusions can be specified (or they will use the defaults).

#         Parameters:
#         -----------
#         ave_type: str
#             The type of average to use. Options are 'triangle', 'vwa', 'simple', 
#             and 'medial'. Default is 'triangle'.
#         n: int
#             The number of periods to use in the average calculation. If None, use 
#             all available periods. If ave_type is 'triangle', this parameter is
#             ignored.
#         tail: float
#             The tail factor to use in the average calculation. Default is 1.0, or 
#             no tail factor. If ave_type is 'triangle', this parameter is ignored.
#         excludes: str
#             The exclusions to use in the average calculation. Default is 'hl', 
#             or high and low. If ave_type is 'triangle', this parameter is ignored.
#             This parameter is a string of characters, where each character is an
#             exclusion. The options are:
#                 h - high
#                 l - low
#                 m - median
#             These characters can be in any order, and any number of them can be
#             specified. For example, 'hl' excludes the high and low values, as does
#             'lh', but 'hhl' excludes only the high value.
                
            
#         Returns:
#         --------
#         ata: pd.DataFrame
#             The age-to-age factors of the triangle data, depending on the average
#             type. Shape is the same as the triangle data.
#         """
#         # if the average type is 'triangle', return the triangle of age-to-age factors
#         if ave_type.lower() == 'triangle':
#             return self._ata_tri()
#         # if the average type is 'vwa', return the volume weighted average age-to-age factors
#         elif ave_type.lower() == 'vwa':
#             return self._vwa(n=n, tail=tail)
#         # if the average type is 'simple', return the simple average age-to-age factors
#         elif ave_type.lower() == 'simple':
#             return self._ave_ata(n=n, tail=tail)
#         # if the average type is 'medial', return the medial average age-to-age factors
#         elif ave_type.lower() == 'medial':
#             return self._medial_ata(n=n, tail=tail, excludes=excludes)
#         # if the average type is not recognized, raise an error
#         else:
#             raise ValueError('Invalid age-to-age type. Must be "triangle", "vwa", "simple", or "medial"')
        
#     def atu(self,
#             ave_type: str = 'vwa',
#             n: int = None,
#             tail: float = 1.0,
#             excludes: str = 'hl',
#             custom: np.ndarray = None
#             ) -> pd.DataFrame:
#         """
#         Calculates the age-to-ultimate factors from the triangle data.

#         Parameters:
#         -----------
#         ave_type: str
#             The type of average to use. Options are 'vwa', 'simple', 
#             and 'medial'. Default is 'vwa'.
#         n: int
#             The number of periods to use in the average calculation. If None, use 
#             all available periods. 
#         tail: float
#             The tail factor to use in the average calculation. Default is 1.0, or 
#             no tail factor.
#         excludes: str
#             The exclusions to use in the average calculation. Default is 'hl', 
#             or high and low. This parameter is a string of characters, where each 
#             character is an exclusion. The options are:
#                 h - high
#                 l - low
#                 m - median
#             These characters can be in any order, and any number of them can be
#             specified. For example, 'hl' excludes the high and low values, as does
#             'lh', but 'hhl' excludes only the high value.
#         custom: np.ndarray
#             A custom array of age-to-age factors to use in the calculation. If
#             None, use the age-to-age factors calculated from the 'ave_type'.
#             If not None, the 'ave_type', 'n', 'tail', and 'excludes' parameters
#             are ignored.
#             Default is None.

#         Returns:
#         --------
#         atu: pd.DataFrame
#             The age-to-ultimate factors of the triangle data.
#         """
#         # calculate the age-to-age factors
#         if custom is None:
#             age_to_age = self.ata(ave_type=ave_type, n=n, tail=tail, excludes=excludes)
#         else:
#             age_to_age = pd.Series(custom, index=self.tri.columns)

#         # calculate the age-to-ultimate factors (cumulative product of the ata factors, 
#         # starting with the last column/the tail factor)
#         age_to_ult = age_to_age[::-1].cumprod()[::-1]

#         return age_to_ult
    
#     def diag(self, calendar_year: int = None) -> pd.DataFrame:
#         """
#         Calculates the specified diagonal of the triangle data.

#         Parameters:
#         -----------
#         calendar_year: int
#             The calendar year of the diagonal to return. If None, return the
#             current diagonal. Default is None.
#             This is not implemented.

#         Returns:
#         --------
#         diag: pd.DataFrame
#             The diagonal of the triangle data.
#         """
#         # look at the triangle as an array
#         triangle_array = self.tri.to_numpy()



#         # if the calendar year is not specified, return the current diagonal
#         if calendar_year is None:
#             calendar_year = triangle_array.shape[0]

#         # diagonal is a series of length equal to the number of rows in the triangle
#         diag = pd.Series(np.diagonal(np.fliplr(triangle_array)), index=self.tri.index)

#         # match the index of the diagonal to column name that value can be found in
#         # (remember that the triangle may not be the same size as the index, if the
#         # triangle is not square -- so we need to actually match the first occurrence
#         # of the value to the column name)
# ################################################################################################################################################################




#         return diag
    
#     def summary(self,
#                 ave_type: str = 'vwa',
#                 n: int = None,
#                 tail: float = 1.0,
#                 excludes: str = 'hl',
#                 custom: pd.Series = None
#                 ) -> pd.DataFrame:
#         """
#         Produces a summary table that includes the origin period, development period,
#         cumulative loss, age-to-ultimate factors, and ultimate loss.
#         """
#         # if custom selection is specified, use that, otherwise use the selection
#         if custom is None:
#             age_to_ult = self.atu(ave_type=ave_type, n=n, tail=tail, excludes=excludes)
#         else:
#             age_to_ult = self.atu(custom=custom)
        
#         # reverse the age-to-ultimate factors so they match up with the loss
#         age_to_ult = age_to_ult.iloc[::-1]

#         # calculate the cumulative loss
#         cum_loss = self.diag()

#         # output data frame
#         df = pd.DataFrame(dict(ay=self.tri.index, cum_loss=cum_loss.values, age_to_ult=age_to_ult.values))
        
#         # calculate the ultimate loss
#         df['ult_loss'] = df['cum_loss'] * df['age_to_ult']

#         df.columns = ['AY', 'Cumulative', 'Age-to-Ult', 'Ultimate']

#         # display "Cumulative" and "Ultimate" with commas and no decimal places
#         df['Cumulative'] = df['Cumulative'].apply(lambda x: formatter(x, 'dollars'))
#         df['Ultimate'] = df['Ultimate'].apply(lambda x: formatter(x, 'dollars'))

#         # display Age-to-Ult with 3 decimal places
#         df['Age-to-Ult'] = df['Age-to-Ult'].apply(lambda x: formatter(x, 'factors'))

#         # set the index to the accident year
#         df = df.set_index('AY')

#         return df                    
    
# ################################################################################################################################################################
# ########## Needed for compatibility with the other classes in the package. ###########
# ################################################################################################################################################################
#     def fit(self, X, y):
#         raise NotImplementedError("This method should be implemented in the child class.")

#     def Predict(self, X):
#         raise NotImplementedError("This method should be implemented in the child class.")

#     def ata_summary(self) -> pd.DataFrame:

#         triangle = self#.tri

#         ata_tri = triangle.ata().round(3)

#         vol_wtd  = pd.DataFrame({
#             'Vol Wtd': pd.Series(["" for _ in range(ata_tri.shape[1]+1)], index=ata_tri.columns)
#             , 'All Years': triangle.ata('vwa').round(3)
#             , '5 Years': triangle.ata('vwa', 5).round(3)
#             , '3 Years': triangle.ata('vwa', 3).round(3)
#             , '2 Years': triangle.ata('vwa', 2).round(3)
#             }).transpose()

#         simple = pd.DataFrame({
#             'Simple': pd.Series(["" for _ in range(ata_tri.shape[1]+1)], index=ata_tri.columns),
#             'All Years': triangle.ata('simple').round(3),
#             '5 Years': triangle.ata('simple', 5).round(3),
#             '3 Years': triangle.ata('simple', 3).round(3),
#             '2 Years': triangle.ata('simple', 2).round(3)}).transpose()

#         medial = pd.DataFrame({
#             'Medial 5-Year': pd.Series(["" for _ in range(ata_tri.shape[1]+1)], index=ata_tri.columns),
#             'Ex. Hi/Low': triangle.ata('medial', 5, excludes='hl').round(3),
#             'Ex. Hi': triangle.ata('medial', 5, excludes='h').round(3),
#             'Ex. Low': triangle.ata('medial', 5, excludes='l').round(3),
#                                 }).transpose()

#         out = (pd.concat([ata_tri.drop(index=10), vol_wtd, simple, medial], axis=0)
#                .drop(columns=self.tri.columns[-1])
#                .fillna(''))
        
#         return out      
