import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge

from typing import Union

from triangle import Triangle
from BaseEstimator import BaseEstimator


class PaidIncurredChain(BaseEstimator):
    def __init__(self
                 , paid_triangle: Triangle = None
                 , incurred_triangle: Triangle = None
                 , bayesian: bool = False
                 , prior_phi: np.ndarray = None
                 , prior_psi: np.ndarray = None
                 , prior_sigma2: np.ndarray = None
                 , prior_tau2: np.ndarray = None
                 , prior_s2: np.ndarray = None
                 , prior_t2: np.ndarray = None
                 ) -> None:
        """
        Initialize the PaidIncurredChain class with the given triangles of
        claims payments and incurred losses.

        Args:
            paid_triangle (Triangle): 
                The claims payments triangle. Rows represent accident years,
                and columns represent development years. Each element represents
                the cumulative claims payments for a specific accident year and
                development year. 
            incurred_triangle (Triangle):
                The incurred losses triangle. Rows represent accident years, and
                columns represent development years. Each element represents the
                cumulative incurred losses for a specific accident year and
                development year.
            bayesian (bool):
                Whether to use the full Bayesian version of the model. If False,
                the variance parameters are treated as fixed and identical to
                the prior parameters. Default is False.
            prior_phi (np.ndarray):
                The prior for the phi (log paid LDF mean) parameters. If an array
                is passed, the prior for each development period is specified.
                Default is None, in which case the prior parameters are estimated
                from the data.
            prior_psi (np.ndarray):
                The prior for the psi (log incurred LDF mean) parameters. If an array
                is passed, the prior for each development period is specified.
                Default is None, in which case the prior parameters are estimated
                from the data.
            prior_sigma2 (np.ndarray):
                The prior for the sigma2 (log paid LDF variance) parameters. If an array
                is passed, the prior for each development period is specified.
                Default is None, in which case the prior parameters are estimated
                from the data.
            prior_tau2 (np.ndarray):
                The prior for the tau2 (log incurred LDF variance) parameters.
                If an array is passed, the prior for each development period is
                specified. Default is None, in which case the prior parameters are
                estimated from the data.
            prior_s2 (np.ndarray):
                The prior for the s2 (variance of mean log paid LDF) parameters.
                If an array is passed, the prior for each development period is
                specified. Default is None, in which case the prior parameters
                are assumed to be 10 at each development period, effectively placing
                no weight on the prior estimate of the prior mean of the log paid LDF. 
            prior_t2 (np.ndarray):
                The prior for the t2 (variance of mean log incurred LDF) parameters.
                If an array is passed, the prior for each development period is
                specified. Default is None, in which case the prior parameters
                are assumed to be 10 at each development period, effectively placing
                no weight on the prior estimate of the prior mean of the log incurred
                LDF.
        """
        super().__init__()
        # store the paid and incurred triangles in their rocky triangle format
        self.paid_triangle = paid_triangle
        self.incurred_triangle = incurred_triangle

        # use shorter variable names for the paid and incurred triangles as data frames
        self.P = paid_triangle.tri.copy()
        self.I = incurred_triangle.tri.copy()

        # number of development years
        self.J = self.P.shape[1]

        # number of accident years
        self.M = self.P.shape[0]

        # load model prior parameters or estimate them from the data
        if prior_phi is None:
            self.prior_phi = self.empirical_phi_mean()
        else:
            self.prior_phi = prior_phi

        if prior_psi is None:
            self.prior_psi = self.empirical_psi_mean()
        else:
            self.prior_psi = prior_psi

        if prior_sigma2 is None:
            self.prior_sigma2 = self.empirical_sigma2_estimator()
        else:
            self.prior_sigma2 = prior_sigma2

        if prior_tau2 is None:
            self.prior_tau2 = self.empirical_tau2_estimator()
        else:
            self.prior_tau2 = prior_tau2

        if prior_s2 is None:
            self.prior_s2 = np.repeat(10, self.J)
        else:
            self.prior_s2 = prior_s2

        if prior_t2 is None:
            self.prior_t2 = np.repeat(10, self.J-1)
        else:
            self.prior_t2 = prior_t2

        # initialize the model parameters
        self.phi = self.prior_phi
        self.psi = self.prior_psi
        self.sigma2 = self.prior_sigma2
        self.tau2 = self.prior_tau2
        self.s2 = self.prior_s2
        self.t2 = self.prior_t2

    def _para_tbl(self, param: str) -> pd.DataFrame:
        """
        Helper function for ParameterTable
        """
        out = pd.DataFrame()

        if param == "phi":
            out["prior"] = self.prior_phi
            out["posterior"] = self.phi
            out['parameter'] = "phi"
            out['age'] = self.P.columns[:len(self.prior_phi)]
        elif param == "psi":
            out["prior"] = self.prior_psi
            out["posterior"] = self.psi
            out['parameter'] = "psi"
            out['age'] = self.P.columns[:len(self.prior_psi)]
        elif param == "sigma2":
            out["prior"] = self.prior_sigma2
            out["posterior"] = self.sigma2
            out['parameter'] = "sigma2"
            out['age'] = self.P.columns[:len(self.prior_sigma2)]
        elif param == "tau2":
            out["prior"] = self.prior_tau2
            out["posterior"] = self.tau2
            out['parameter'] = "tau2"
            out['age'] = self.P.columns[:len(self.prior_tau2)]
        elif param == "s2":
            out["prior"] = self.prior_s2
            out["posterior"] = self.s2
            out['parameter'] = "s2"
            out['age'] = self.P.columns[:len(self.prior_s2)]
        elif param == "t2":
            out["prior"] = self.prior_t2
            out["posterior"] = self.t2
            out['parameter'] = "t2"
            out['age'] = self.P.columns[:len(self.prior_t2)]
        else:
            raise ValueError("Invalid parameter name.")

        out = out[['parameter', 'age', 'prior', 'posterior']]

        return out

    def ParameterTable(self, param: Union[str, list] = None
                       ) -> pd.DataFrame:
        """
        Returns a table with the prior and posterior parameter estimates.

        Parameters
        ----------
        param : str or list
            The parameter(s) for which to return the table. If None,
            all parameters are returned. Default is None.

        Returns
        -------
        out : pd.DataFrame
            A table with the prior and posterior parameter estimates.
        """
        out = pd.DataFrame()

        if param is None:
            for p in ["phi", "psi", "sigma2", "tau2", "s2", "t2"]:
                out = pd.concat([out, self._para_tbl(p)])
        elif isinstance(param, str):
            out = self._para_tbl(param)
        elif isinstance(param, list):
            for p in param:
                if p in ["phi", "psi", "sigma2", "tau2", "s2", "t2"]:
                    out = pd.concat([out, self._para_tbl(p)])
                else:
                    Warning(f"Invalid parameter name: {p}")
                    pass
        else:
            raise ValueError("Invalid parameter name.")

        return out

    #########################################################
    # this section reproduces the calculations in Section 2 #
    #########################################################

    def _paid_ldf(self, i: int, j: int
                  ) -> float:
        """
        Convenience function to calculate the paid loss development factor
        at the given row and column indices.
        """
        if j == 0:
            return self.P.iloc[i, j]
        else:
            return self.P.iloc[i, j] / self.P.iloc[i, j - 1]

    def paid_ldf(self) -> np.ndarray:
        """
        Convenience function to calculate the paid loss development
        factors.
        """
        P = np.array(self.P)
        P_shifted = np.hstack((np.zeros((P.shape[0], 1)), P[:, :-1]))
        LDF = np.divide(P, P_shifted, out=np.ones_like(P),
                        where=(P_shifted != 0))
        LDF[:, 0] = P[:, 0]
        return LDF

    def _incurred_ldf(self, i: int, j: int
                      ) -> float:
        """
        Convenience function to calculate the incurred loss development factor
        at the given row and column indices.
        """
        if j == 0:
            return self.I.iloc[i, j]
        else:
            return self.I.iloc[i, j] / self.I.iloc[i, j - 1]

    def incurred_ldf(self) -> np.ndarray:
        """
        Vectorized function to calculate the incurred loss development factor matrix.
        """
        I = np.array(self.I)
        I_shifted = np.hstack((np.zeros((I.shape[0], 1)), I[:, :-1]))
        LDF = np.divide(I, I_shifted, out=np.ones_like(I),
                        where=(I_shifted != 0))
        LDF[:, 0] = I[:, 0]
        return LDF

    def _mu(self, j: int
            ) -> float:
        """
        v^2 parameter for incurred log losses conditional on data in
        the incurred triangle only.

        Given in Proposition 2.2 in the paper. 

        Calculated as:

        (sum from 0 to J of the phi values)
        -
        (sum from j to J-1 of the psi values)

        """
        if self.psi is None:
            raise ValueError("The psi values have not been calculated yet.")
        if self.phi is None:
            raise ValueError("The phi values have not been calculated yet.")

        # sum of phi values
        phi_sum = np.sum(self.phi)

        # sum of psi values
        psi_sum = np.sum(self.psi[j:])

        # return the mu value
        if j == self.J:
            return phi_sum
        else:
            return phi_sum - psi_sum

    def ldf_estimator(self, triangle: str = 'paid') -> np.ndarray:
        """
        Estimate of the LDFs for the given triangle.

        From equation 5.2 in the textbook.

        Calculated as:

        ldf_{i, j} = 
        (sum from i=0 to I-j of {log [ C_{i,j} / C_{i, j-1} })
        /
        (I - j + 1)
        """
        if triangle == "paid":
            C = self.P
        elif triangle == "incurred":
            C = self.I
        else:
            raise ValueError(
                f"Invalid triangle: {triangle}. Must be 'paid' or 'incurred'.")

        num_columns = C.shape[1]
        ldf = np.zeros(num_columns - 1)

        for j in range(1, num_columns):
            sum_log_ratios = 0
            count = 0

            for i in range(C.shape[0] - j):
                if not np.isnan(C.iloc[i, j]) and not np.isnan(C.iloc[i, j - 1]) and C.iloc[i, j - 1] != 0:
                    sum_log_ratios += np.log(C.iloc[i, j] / C.iloc[i, j - 1])
                    count += 1

            if count > 0:
                ldf[j - 1] = sum_log_ratios / count

        return ldf

    def variance_estimator(self, triangle: str = "paid") -> float:
        """
        Estimate of the sigma2 parameter or the tau2 parameter for paid
        or incurred log losses conditional on data in the paid or incurred
        triangle only.

        From equation 5.3 in the textbook.  

        Calculated as:

        (sum from i=0 to I-j of {log(LDF_i,j) - log(estimated LDF_j)}^2)
        /
        (I - j)

        where estimated LDF_{i, j} is the estimated LDF for the given row and
        column indices, and is given by:

        """
        if triangle == "paid":
            LDF = self.paid_ldf()[:, 1:]
            tri = self.paid_triangle
        elif triangle == "incurred":
            LDF = self.incurred_ldf()[:, 1:]
            tri = self.incurred_triangle
        else:
            raise ValueError(
                "The triangle argument must be either 'paid' or 'incurred'.")

        logLDF = np.log(LDF)
        logLDF_est = self.ldf_estimator(triangle)

        # for each column, replace nan values with 0, calculate the sum of the squared differences
        # between each nonzero item and the estimated LDF for that column
        # should return a vector of length LDF.shape[1]
        diff2 = np.sum(np.square(np.where(np.logical_or(
            np.isnan(logLDF), np.equal(logLDF, 0)), 0, logLDF - logLDF_est)), axis=0)

        variance = diff2 / (tri.n_rows - np.arange(1, LDF.shape[1] + 1))

        # note this does not provide an estimator for the last column, so we will use loglinear
        # regression with regularization to estimate the variance for the last column
        y = np.log(variance[:-1])
        X = np.arange(1, len(y) + 1).reshape(-1, 1)
        model = Ridge(alpha=1).fit(X, y)

        # calculate the variance for the last column
        last_col_var = np.exp(model.predict([[len(y) + 1]]))

        # return the variance vector with the last column variance appended
        return np.append(variance[:-1], last_col_var)

    def empirical_sigma2_estimator(self) -> np.ndarray:
        """
        Prior estimate of the sigma2 parameter for paid losses conditional on data in the paid
        triangle only. Uses the textbook standard deviation estimator.

        This is an alias for the variance_estimator function with the triangle argument set to
        "paid".
        """
        return self.variance_estimator(triangle="paid")

    def empirical_tau2_estimator(self) -> np.ndarray:
        """
        Prior estimate of the tau2 parameter for incurred losses conditional on data in the
        incurred triangle only. Uses the textbook standard deviation estimator.

        This is an alias for the variance_estimator function with the triangle argument set to
        "incurred", but does not include the final column, since ultimate incurred is set equal
        to ultimate paid.
        """
        return self.variance_estimator(triangle="incurred")[:-1]

    def mu(self) -> np.ndarray:
        """
        Vectorized function to calculate the mu values for all columns j.

        This function calculates the mu values for all columns j by creating an array of the same
        length as the number of columns in the incurred triangle. It calculates the sum of phi
        values and subtracts the cumulative sum of psi values for each j. The last element in the
        array remains the sum of phi values.
        """
        if self.psi is None:
            raise ValueError("The psi values have not been calculated yet.")
        if self.phi is None:
            raise ValueError("The phi values have not been calculated yet.")

        J = self.J

        # sum of phi values
        phi_sum = np.sum(self.phi)

        # sum of psi values for each j
        psi_cumsum = np.cumsum(self.psi[::-1])[::-1]

        # calculate mu values for all j
        mu_values = np.full(J + 1, phi_sum)
        mu_values[:-1] -= psi_cumsum

        return mu_values

    def _v2(self, j: int) -> float:
        """
        v^2 parameter for incurred log losses conditional on data in
        the incurred triangle only.

        Given in Proposition 2.2 in the paper. 

        Calculated as:

        (sum from 0 to J of the sigma squared values)
        +
        (sum from j to J-1 of the tau squared values)

        """
        if self.sigma2 is None:
            raise ValueError("The sigma2 values have not been calculated yet.")
        if self.tau2 is None:
            raise ValueError("The tau2 values have not been calculated yet.")

        # sum of sigma squared values
        sigma2_sum = np.sum(self.sigma2)

        # sum of tau squared values
        tau2_sum = np.sum(self.tau2[j:])

        # return the v2 value
        if j == self.J:
            return sigma2_sum
        else:
            return sigma2_sum + tau2_sum

    def v2(self, t2: np.ndarray = None) -> np.ndarray:
        """
        Vectorized function to calculate the v^2 values for all columns j.

        Takes an optional t2 array as an argument. If t2 is provided, the function
        uses it instead of tau2 to calculate the v^2 values for all columns j.
        """
        J = self.J

        # sum of sigma squared values
        sigma2_sum = np.sum(self.sigma2)

        # use t2 if provided, otherwise use tau2
        t2 = self.tau2 if t2 is None else t2

        # sum of t2 squared values for each j
        t2_cumsum = np.cumsum(t2[::-1])[::-1]

        # calculate v2 values for all j
        v2_values = np.full(J + 1, sigma2_sum)
        v2_values[:-1] += t2_cumsum

        return v2_values

    def _cond_log_incurred_mean(self, i: int, j: int, l: int
                                ) -> float:
        """
        Mean of the conditional distribution of the log incurred losses
        conditional on data in the incurred triangle only.

        Parameters
        ----------
        i : int
            The row index of the incurred losses triangle.
        j : int
            The column index of the incurred losses triangle that represents
            the current/most recent development year.
        l : int
            The number of development years in the future. This is a general
            formulation of the mean, and takes the current triangle, but allows
            estimation of any future development year.

        Returns
        -------
        float
            The mean of the conditional distribution of the log incurred losses
            conditional on data in the incurred triangle only.

        References
        ----------
        Given in Proposition 2.2 in the paper.

        Calculation
        -----------
        $$E[\log I_{i, j+l} | D_j^I, \Phi] = \mu(j+l) + \frac{v^2(j+l)}{v^2(j)} (\log I_{i, j} - \mu(j))$$
        """
        return self._mu(j + l) + (self._v2(j + l) / self._v2(j)) * (np.log(self.I.iloc[i, j]) - self._mu(j))

    def cond_log_incurred_mean(self) -> np.ndarray:
        """
        Vectorized function to calculate the mean of the conditional distribution 
        of the log incurred losses for all combinations of i, j, and l.
        """
        # Obtain the shape of the incurred triangle (I is the number of rows, J is the number of columns)
        I, J = self.I.shape

        # Calculate the mu values for all columns j using the vectorized mu() method
        mu_values = self.mu()

        # Calculate the v2 values for all columns j using the vectorized v2() method
        v2_values = self.v2()

        # Reshape mu_values to have shape (J+1, 1) for broadcasting during calculations
        mu_j_plus_l = mu_values[:, np.newaxis]

        # Calculate the ratio of v2 values for all pairs of (j+l, j) where v2(j) is not zero
        # The result has shape (J+1, J+1)
        v2_ratio = np.divide(
            v2_values[:, np.newaxis], v2_values, where=(v2_values != 0))

        # Calculate the difference between the logarithm of the incurred triangle (element-wise)
        # and the mu_values (except the last one), broadcasting mu_values along the rows
        # The result has shape (I, J)
        log_I_minus_mu_j = np.log(self.I) - mu_values[np.newaxis, :-1]

        # Calculate the product of v2_ratio and log_I_minus_mu_j
        # The result has shape (J+1, J+1, I, J)
        product_term = v2_ratio[:, :, np.newaxis] * \
            log_I_minus_mu_j[np.newaxis, np.newaxis, :, :]

        # Add mu_j_plus_l to the product_term, broadcasting mu_j_plus_l along the last two dimensions
        # The result has shape (J+1, J+1, I, J)
        cond_mean = mu_j_plus_l[:, :, np.newaxis, np.newaxis] + product_term

        return cond_mean

    def _cond_log_incurred_var(self, i: int, j: int, l: int) -> float:
        """
        Variance of the conditional distribution of the log incurred losses
        conditional on data in the incurred triangle only.

        Parameters
        ----------
        i : int
            The row index of the incurred losses triangle.
        j : int
            The column index of the incurred losses triangle that represents
            the current/most recent development year.
        l : int
            The number of development years in the future. This is a general
            formulation of the variance, and takes the current triangle, but allows
            estimation of any future development year.

        Returns
        -------
        float
            The variance of the conditional distribution of the incurred losses
            conditional on data in the incurred triangle only.

        References
        ----------
        Given in Proposition 2.2 in the paper.

        Calculation
        -----------
        $$\text{Var}[\log I_{i, j+l} | D_j^I, \Phi] = v_{j+l}^2 \left(1 - \frac{v_{j+l}^2}{v_{j}^2} \right)$$
        """
        return self._v2(j + l) * (1 - (self._v2(j + l) / self._v2(j)))

    def cond_log_incurred_var(self) -> np.ndarray:
        """
        Vectorized function to calculate the variance of the conditional distribution 
        of the log incurred losses for all combinations of i, j, and l.
        """
        # Obtain the shape of the incurred triangle (I is the number of rows, J is the number of columns)
        I, J = self.I.shape

        # Calculate the v2 values for all columns j using the vectorized v2() method
        v2_values = self.v2()

        # Calculate the ratio of v2 values for all pairs of (j+l, j) where v2(j) is not zero
        # The result has shape (J+1, J+1)
        v2_ratio = np.divide(
            v2_values[:, np.newaxis], v2_values, where=(v2_values != 0))

        # Calculate 1 - v2_ratio, ensuring that the result is 0 where v2(j) is zero
        # The result has shape (J+1, J+1)
        one_minus_v2_ratio = np.where(v2_values != 0, 1 - v2_ratio, 0)

        # Calculate the product of v2_values (j+l) and one_minus_v2_ratio
        # The result has shape (J+1, J+1)
        var_values = v2_values[:, np.newaxis] * one_minus_v2_ratio

        # Add two new dimensions to var_values to have shape (J+1, J+1, 1, 1)
        # This allows broadcasting along the i and j dimensions
        var_values_expanded = var_values[:, :, np.newaxis, np.newaxis]

        # Broadcast var_values_expanded along the i and j dimensions to create a 3D array
        # with shape (J+1, J+1, I, J) containing the variances for all combinations of i, j, and l
        cond_var = np.broadcast_to(var_values_expanded, (J+1, J+1, I, J))

        return cond_var

    def _alpha(self, j: int) -> float:
        """
        Alpha credibility weight for expected ultimate losses conditional on
        data in the incurred triangle only, and parameter vector Theta.

        Given in Corollary 2.3 of the paper.

        Calculated as:

        1 - (_v2(J) / _v2(j))
        """
        return 1 - (self._v2(self.J) / self._v2(j))

    def alpha(self) -> np.ndarray:
        """
        Vectorized function to calculate the alpha credibility weights for all j.
        """
        # Calculate the v2 values for all columns j using the vectorized v2() method
        v2_values = self.v2()

        # Calculate the ratio of v2 values for all pairs of (J, j) where v2(j) is not zero
        # The result has shape (J+1,)
        v2_ratio = np.divide(v2_values[-1], v2_values, where=(v2_values != 0))

        # Calculate 1 - v2_ratio, ensuring that the result is 0 where v2(j) is zero
        # The result has shape (J+1,)
        alpha_values = np.where(v2_values != 0, 1 - v2_ratio, 0)

        return alpha_values

    def _cond_incurred_mean_ultimate(self, i: int, j: int
                                     ) -> float:
        """
        Mean of the conditional distribution of the incurred ultimate losses
        conditional on data in the incurred triangle only.

        Given in Corollary 2.3 of the paper.

        Calculated as:

        I_{i,j}
        *
        exp[(sum from j to J-1 of {psi values + (tau squared values)/2})]
        *
        exp[alpha(j) * (mu(j) - log(I_{i,j}) - (sum from j to J-1 of {tau squared values}/2))]
        """
        # sum of psi values
        psi_sum = np.sum(self.psi[j:])

        # sum of tau squared / 2 values
        tau2_sum = np.sum(self.tau2[j:]) / 2

        # other values
        Iij = self.I.iloc[i, j]
        alpha_j = self._alpha(j)
        mu_j = self._mu(j)

        # return the mean
        return Iij * np.exp(psi_sum + tau2_sum) * np.exp(alpha_j * (mu_j - np.log(Iij) - tau2_sum))

    def _eta(self, j: int) -> float:
        """
        Eta parameter for log paid losses at ultimate conditional on data in
        both paid and incurred triangles.

        Given in Theorem 2.4 of the paper. 

        Calculated as:

        (sum from 0 to j of the phi values)

        """
        # sum of phi values
        phi_sum = np.sum(self.phi[:j+1])

        # return the eta value
        return phi_sum

    def _w2(self, j: int) -> float:
        """
        w^2 parameter for log paid losses at ultimate conditional on data in
        both paid and incurred triangles.

        Given in Theorem 2.4 of the paper. 

        Calculated as:

        (sum from 0 to j of the sigma squared values)

        """
        # sum of sigma squared values
        sigma2_sum = np.sum(self.sigma2[:j+1])

        # return the w2 value
        return sigma2_sum

    def _beta(self, j: int) -> float:
        """
        Beta credibility parameter for log paid losses at ultimate conditional
        on data in both paid and incurred triangles.

        Given in Theorem 2.4 of the paper. 

        Calculated as:

        [v2(J) - w2(j)]
        /
        [v2(j) - w2(j)]
        """
        return (self._v2(self.J) - self._w2(j)) / (self._v2(j) - self._w2(j))

    def _cond_log_paid_mean_ultimate(self, i: int, j: int
                                     ) -> float:
        """
        Mean of the conditional distribution of the log paid ultimate losses
        for origin period i, conditional on data in both paid and incurred
        triangles through development year j.

        Given in Theorem 2.4 of the paper.

        Calculated as:

        mu(J)
        +
        (1 - beta(j)) * (log P_{i,j} - eta(j))
        +
        beta(j) * (log I_{i,j} - mu(j))
        """
        # other values
        mu_J = self._mu(self.J)
        beta_j = self._beta(j)
        eta_j = self._eta(j)
        P_ij = self.P.iloc[i, j]
        I_ij = self.I.iloc[i, j]
        mu_j = self._mu(j)

        # return the mean
        cond_mean = mu_J
        cond_mean += (1 - beta_j) * (np.log(P_ij) - eta_j)
        cond_mean += beta_j * (np.log(I_ij) - mu_j)

        return cond_mean

    def _cond_log_paid_var_ultimate(self, i: int, j: int
                                    ) -> float:
        """
        Variance of the conditional distribution of the log paid ultimate
        losses for origin period i, conditional on data in both paid and
        incurred triangles through development year j.

        Given in Theorem 2.4 of the paper.

        Calculated as:

        (1 - beta(j)) * (v2(J) - w2(j))
        """
        # other values
        beta_j = self._beta(j)
        v2_J = self._v2(self.J)
        w2_j = self._w2(j)

        # return the variance
        return (1 - beta_j) * (v2_J - w2_j)

    def _pic_ultimate_loss_chain_ladder_adjustment(self, i: int, j: int
                                                   ) -> float:
        """
        Adjustment factor applied to the standard chain ladder ultimate
        loss based on paid data only. This adjustment factor compares the 
        incurred-paid ratios and corresponds to the observed residuals:

        log[I_{i,j} / P_{i,j}] - (mu(j) - eta(j))

        A large incurred-paid ratio will result in a large positive adjustment
        to the classical chain ladder ultimate loss predictor. 

        Note that this is a similar adjustment to the one used in the
        Munich chain ladder method that also uses the observed incurred-paid
        ratios to adjust the chain ladder ultimate loss predictor.

        Given as a remark to Theorem 2.4 of the paper.

        Calculated as (numbered steps correspond to the numbered equations
        in the calculation below):

        (1) exp[
        (2)     beta(j)
                *
                (
        (3)         log [I_{i,j} / P_{i,j}]
                    -
        (4)         (mu(j) - eta(j))
                    - 
        (5)         (sum from j+1 to J of sigma^2/2 values)
                )
            ]
        """
        # other values
        beta_j = self._beta(j)
        I_ij = self.I.iloc[i, j]
        P_ij = self.P.iloc[i, j]
        mu_j = self._mu(j)
        eta_j = self._eta(j)
        sigma2_sum = np.sum(self.sigma2[j+1:])

        # calculate the adjustment
        cl_adjustment = np.log(I_ij / P_ij)   # (3)
        cl_adjustment -= (mu_j - eta_j)       # (4)
        cl_adjustment -= (sigma2_sum / 2)     # (5)
        cl_adjustment *= beta_j               # (2)
        cl_adjustment = np.exp(cl_adjustment)  # (1)

    def _pic_ultimate_loss_prediction(self, i: int, j: int
                                      ) -> float:
        """
        Predicted ultimate loss for origin period i, conditional on data in
        both paid and incurred triangles through development year j, and 
        parameter vector Theta.

        Given in Corollary 2.5 of the paper.

        Calculated as:

        (1) P_{i,j}
            *
        (2) exp[
        (3)     (sum from j+1 to J of phi_j values)
                +
        (4)     (sum from j+1 to J of sigma^2/2 values)
            ]
            *
        (5) adjustment factor
        """
        # other values
        P_ij = self.P.iloc[i, j]
        phi_sum = np.sum(self.phi[j+1:])
        sigma2_sum = np.sum(self.sigma2[j+1:])
        cl_adjustment = self._pic_ultimate_loss_chain_ladder_adjustment(i, j)

        # calculate the prediction step by step
        pred = phi_sum + (sigma2_sum / 2)  # (3) + (4)
        pred = np.exp(pred)               # (2)
        pred *= P_ij                      # (1)
        pred *= cl_adjustment             # (5)

        return pred

    #########################################################
    # This section reproduces the calculations in section 3 #
    #########################################################

    def _hash(self, j: int
              ) -> int:
        """
        Hash (#) function from the paper. This is the number of
        future development years. See Theorem 3.2 in the paper.
        """
        return self.J - j + 1

    def n_future_development_years(self):
        """
        Number of future development years. See Theorem 3.2 in the paper.
        """
        return pd.Series(self._hash(j) for j in range(self.J)).to_numpy()

    def _gammaP(self, j: int
                ) -> float:
        """
        GammaP credibility weight for the posterior distribution of
        phi. See Theorem 3.2 in the paper.

        Calculated as:
        hash(j)
        / 
        (
            hash(j)
            +
            sigma^2_j / s^2_j
        )
        """
        sigma2_j = self.sigma2[j]
        s2_j = self.s2[j]
        return self._hash(j) / (self._hash(j) + (sigma2_j / s2_j))

    def gammaP(self) -> np.ndarray:
        """
        Credibility weights for the posterior distribution of phi.
        See Theorem 3.2 in the paper.
        """
        denom = self.n_future_development_years(
        )[:-1] + (np.divide(self.sigma2, self.s2))
        num = self.n_future_development_years()[:-1]
        return np.divide(num, denom)

    def _empirical_phi_mean(self, j: int
                            ) -> float:
        """
        Empirical mean of the posterior distribution of phi_j.

        Given by Theorem 3.2 in the paper.

        Calculated as:
        (1) sum of the log paid LDFs from 0 to J - j
        /
        (2) hash(j)
        """
        paid_ldfs = [self._paid_ldf(i) for i in range(self.J - j + 1)]
        return np.sum(np.log(paid_ldfs)) / self._hash(j)

    def empirical_phi_mean(self) -> np.ndarray:
        """
        Empirical means of the posterior distribution of phi.
        """
        return self.ldf_estimator('paid')

    def posterior_phi_mean(self) -> float:
        """
        Posterior mean of phi_j, given the observed data.

        Given by Theorem 3.2 in the paper.

        Calculated as:

        (1) gammaP(j) * empirical_phi_mean(j)
        +
        (2) (1 - gammaP(j)) * phi(j)
        """
        post = np.multiply(self.gammaP(), self.empirical_phi_mean())
        post += np.multiply((1 - self.gammaP()), self.prior_phi)
        return post

    def _posterior_phi_sample_variance(self, j: int
                                       ) -> float:
        """
        Posterior variance of phi_j, given the observed data.

        Given by Theorem 3.2 in the paper.

        Calculated as:

        1
        / 
        [
            ( 1/ s2(j) )
            +
            ( hash(j) / sigma^2_j )
        ]
        """
        s2_j = self.s2[j]
        sigma2_j = self.sigma2[j]
        return 1 / ((1 / s2_j) + (self._hash(j) / sigma2_j))

    def posterior_phi_sample_variance(self) -> np.ndarray:
        """
        Posterior sample variances of phi.
        """
        one = np.power(self.s2, -1)
        two = np.divide(self.n_future_development_years()[:-1], self.sigma2)
        three = np.add(one, two)
        return np.power(three, -1)

    def _posterior_ultimate_paid_loss(self, i: int
                                      ) -> float:
        """
        Posterior mean of the ultimate loss for origin period i, based
        on the paid loss data.

        Given as Equation 3.1 in the paper.

        Calculated as:

        (1) P_{i, J-i} 
            *
        (2) product from l = J-i+1 to J of [
        (3)     exp[
        (4)         posterior_phi_mean(l)
                    +
        (5)         sigma^2(l) / 2
                    +
        (6)         posterior_phi_sample_variance(l) / 2
                ]
            ]
        """
        P_ij = self.P.iloc[i, self.J - i]
        post_phi_mean = [self._posterior_phi_mean(
            l) for l in range(self.J - i + 1, self.J + 1)]
        sigma2 = [self.sigma2[l] for l in range(self.J - i + 1, self.J + 1)]
        post_phi_var = [self._posterior_phi_sample_variance(
            l) for l in range(self.J - i + 1, self.J + 1)]

        # calculate the prediction step by step
        ultimate = post_phi_mean + (sigma2 / 2)  # (4) + (5)
        ultimate += (post_phi_var / 2)           # (6)
        ultimate = np.exp(ultimate)              # (3)
        ultimate = np.prod(ultimate)             # (2)
        ultimate *= P_ij                         # (1)

        return ultimate

    def posterior_ultimate_paid_loss(self) -> np.ndarray:
        """
        Posterior means of the ultimate paid loss for each origin period.
        """
        P = self.P.iloc[:, :-1]
        posterior_phi_mean = self.posterior_phi_mean()
        sigma2 = self.sigma2
        posterior_phi_var = self.posterior_phi_sample_variance()

        # calculate the prediction step by step
        ultimate = np.add(posterior_phi_mean, (sigma2 / 2))  # (4) + (5)
        ultimate = np.add(ultimate, (posterior_phi_var / 2))  # (6)
        ultimate = np.exp(ultimate)                          # (3)
        ultimate = np.flip(ultimate)              # (2)
        ultimate = np.cumprod(ultimate)
        ultimate = np.flip(ultimate)              # (2)
        ultimate = np.multiply(P, ultimate)           # (1)

        return ultimate

    def _aI_element(self, n: int, m: int
                    ) -> float:
        """
        Element from a^I matrix at row n and column m. 
        a^I is the inverse of the posterior covariance matrix of psi.

        Given by Theorem 3.3 in the paper. 

        Calculated as:

        (1) sum from j = 0 to min(n,m) of [1 / v2(j)]
            +
            [
                (
        (2)         [1 / t2(n)]
                    +
        (3)         [(J - n) / tau2(n)]
                )
                *
        (4)     1_{n=m}
            ]
        """
        v2 = self.v2[:min(n, m) + 1]
        t2 = self.t2[n]
        tau2 = self.tau2[n]
        J = self.J

        # calculate the element step by step
        a = 1 / t2           # (2)
        a += (J - n) / tau2  # (3)
        a *= (1 == (n == m))  # (4)
        a += np.sum(1 / v2)  # (1)

        return a

    def _aI_matrix(self) -> np.ndarray:
        """
        Posterior covariance matrix of psi.

        Given by Theorem 3.3 in the paper.

        Calculated as:

        (1) a^I_{n,m} = aI_element(n,m)
        """
        aI = np.zeros((self.J + 1, self.J + 1))
        for n in range(self.J + 1):
            for m in range(self.J + 1):
                aI[n, m] = self._aI_element(n, m)
        return aI

    def _gammaI(self, j: int
                ) -> float:
        """
        GammaI credibility weight for the posterior distribution of
        psi. See Theorem 3.3 in the paper.

        Calculated as:
        hash(j) - 1
        / 
        (
            hash(j)
            - 
            1
            +
            tau^2_j / t^2_j
        )
        """
        hash_j = self._hash(j)
        tau2_j = self.tau2[j]
        t2_j = self.t2[j]
        return (hash_j - 1) / (hash_j - 1 + (tau2_j / t2_j))

    def _empirical_psi_mean(self, j: int
                            ) -> float:
        """
        Empirical mean of psi_j, calculated from the observed data.

        Given by Theorem 3.3 in the paper.

        Calculated as:

        (1) sum from i=0 to J-j-1 of (
            log(1 / _incurred_ldf(i, j+1))
            )
            /
        (2) hash(j) - 1
        """
        J = self.J
        hash_j = self._hash(j)
        log_ldf_sum = np.sum(np.log(1 / self._incurred_ldf(i, j + 1))
                             for i in range(J - j))
        return log_ldf_sum / (hash_j - 1)

    def empirical_psi_mean(self) -> np.ndarray:
        """
        Empirical mean of psi_j, calculated from the observed data.
        """
        return self.ldf_estimator('incurred')

    def _credibility_weighted_psi_mean(self, j: int
                                       ) -> float:
        """
        Credibility weighted mean of psi_j, calculated from the observed data.

        Given by Theorem 3.3 in the paper.

        Calculated as:

        (1) gammaI(j) * empirical_psi_mean(j)
            +
        (2) (1 - gammaI(j)) * psi(j)
        """
        gammaI = self._gammaI(j)
        empirical_psi_mean = self._empirical_psi_mean(j)
        psi = self.psi[j]
        return (gammaI * empirical_psi_mean) + ((1 - gammaI) * psi)

    def _bI_element_from_credibility(self, j: int
                                     ) -> float:
        """
        Element j from b^I vector.

        The posterior mean of psi given the data in the incurred losses triangle
        is given by Theorem 3.3 in the paper, and is calculated as:

        Cred-wtd psi mean
        *
        [
            1 / t2(j)
            +
            (
                hash(j)
                -
                1
            )
            /
            tau2(j)
        ]
        """
        psi_j = self._credibility_weighted_psi_mean(j)
        t2_j = self.t2[j]
        tau2_j = self.tau2[j]
        hash_j = self._hash(j)
        return psi_j * ((1 / t2_j) + ((hash_j - 1) / tau2_j))

    def _bI_element(self, j: int
                    ) -> float:
        """
        Element j from b^I vector. 

        The posterior mean of psi given the data in the incurred losses triangle
        is given by Theorem 3.3 in the paper, and is calculated as:

        posterior covariance matrix of psi
        *
        b^I vector

        The b^I vector is given in the theorem as well, and is calculated as:

        (1) psi(j) / t2(j)
            -
            [
        (2)     (sum from i = 0 to J-j-1 of log[_incurred_ldf(i, j)])
                /
        (3)     tau2(j)
            ]
            -
        (4) sum from i = 0 to j of [
                (log I_{J-i, i} / v2(i))
                ]
        """
        psi_j = self.psi[j]
        t2_j = self.t2[j]
        tau2_j = self.tau2[j]
        J = self.J
        ldf = [self._incurred_ldf(i, j) for i in range(J - j)]
        v2 = self.v2[:j + 1]
        logI = [np.log(self.I.iloc[J - i, i]) for i in range(j + 1)]

        # calculate the element step by step
        b = psi_j / t2_j           # (1)
        b -= np.sum(ldf) / tau2_j  # (2) / (3)
        b -= np.sum(logI / v2)     # (4)

        return b

    def _bI(self) -> np.ndarray:
        """
        Convenience function to calculate the b^I vector.
        """
        return np.array([self._bI_element(j) for j in range(self.J + 1)])

    def _likelihood(self) -> float:
        """
        Joint likelihood function of the data D_J and the model parameters psi and v.

        Given by Equation 3.5 in the paper.

        Calculated as:

        (Part A)
            (1) product from j=0 to J of {
            (2)     product from i=0 to J-j of {
            (3)         exp[
            (4)             -(phi(j) - log(_paid_ldf(i, j)))^2
                            /
            (5)             (2 * sigma2(j))
                        ]
                        /
            (6)         P_{i,j} * sqrt(2 * pi * sigma2(j))
                    }
                }
                *
        (Part B)        
            (7)     product from i=1 to J of {
                        [
            (8)             - 1
                            /
            (9)             sqrt(2 * pi * [v2(J-i) - w2(J-i)]) * I_{i, J-i}
                        ]
                    }
                *
        (Part C)
            (10)    exp{
            (11)        -[mu(J-i) - eta(J-i) - log(I_{i, J-i} / P_{i, J-i})]^2
                        /
            (12)        (2 * [v2(J-i) - w2(J-i)])
                    }
                *
        (Part D)
            (13)    product from j=0 to J-1 of {
            (14)        product from i=0 to J-j-1 of {
            (15)            exp[
            (16)                -(psi(j) + log(1 / _incurred_ldf(i, j+1)))^2
                                /
            (17)                (2 * tau2(j))
                            ]
                            /
            (18)            I_{i, j+1} * sqrt(2 * pi * tau2(j))
                    }
        """
        # ================
        # == Part A ======
        # ================

        # Part A-4: -(phi(j) - log(_paid_ldf(i, j)))^2
        phi = self.phi
        paid_ldf = np.array([self._paid_ldf(i, j) for i in range(
            self.J + 1) for j in range(self.J + 1 - i)])
        paid_ldf = paid_ldf.reshape(self.J + 1, self.J + 1)
        paid_ldf = np.log(paid_ldf)
        A4 = np.square(phi - paid_ldf)

        # Part A-5: (2 * sigma2(j))
        sigma2 = self.sigma2
        A5 = 2 * sigma2

        # Part A-3: exp[A4 / A5]
        A3 = np.exp(A4 / A5)

        # Part A-6: P_{i,j} * sqrt(2 * pi * sigma2(j))
        P = self.P
        A6 = P * np.sqrt(2 * np.pi * sigma2)

        # Part A-2: product from i=0 to J-j of {A3 / A6}
        A2 = np.prod(A3 / A6, axis=1)

        # Part A = Part A-1: product from j=0 to J of {A2}
        A = np.prod(A2)

        # ================
        # == Part B ======
        # ================

        # Part B-8: - 1
        B8 = -1

        # Part B-9: sqrt(2 * pi * [v2(J-i) - w2(J-i)]) * I_{i, J-i}
        v2 = self.v2
        w2 = self.w2
        I = self.I
        vct = np.sqrt(2 * np.pi * (v2 - w2))
        # multiply corresponding elements of vct and columns of I
        B9 = vct * I

        # Part B-7: product from i=1 to J of {B8 / B9}
        B7 = np.prod(B8 / B9, axis=0)

        # ================
        # == Part C ======
        # ================

        # Part C-11: -[mu(J-i) - eta(J-i) - log(I_{i, J-i} / P_{i, J-i})]^2
        mu = self.mu
        eta = self.eta
        inc_paid_ratio = np.where(
            # handle division by zero
            self.P == 0,
            # if P_{i, J-i} == 0, set I_{i, J-i} / P_{i, J-i} to 0
            np.zeros(self.I.shape),
            # otherwise, calculate I_{i, J-i} / P_{i, J-i}
            self.I / self.P
        )
        log_inc_paid_ratio = np.log(inc_paid_ratio)

    # def _calculate_log_ratios(self):
    #     """
    #     Calculate the log ratios for both the claims payments and incurred losses triangles. Log ratios are
    #     calculated as the natural logarithm of the ratio of the current element and its adjacent element in the
    #     same row (i.e., the ratio of the current development year and the previous development year).
    #     """
    #     self.logP = self.P.copy()
    #     self.logI = self.I.copy()

    #     # for logP, first column is just the natural log of the first column of P
    #     self.logP.iloc[:, 0] = np.log(self.P.iloc[:, 0])

    #     # for logP, remaining columns are the natural log of the ratio of the current column and the previous column
    #     for col in range(1, self.P.shape[1]):
    #         self.logP.iloc[:, col] = np.where(
    #             np.equal(self.P.iloc[:, col - 1], 0)
    #             , 0
    #             , np.log(self.P.iloc[:, col] / self.P.iloc[:, col - 1])
    #             )

    #     # every column of logI is the natural log of the ratio of the current column (i) and the next column (i + 1)
    #     for col in range(self.I.shape[1] - 1):
    #         self.logI.iloc[:, col] = np.where(
    #             np.equal(self.I.iloc[:, col + 1], 0)
    #             , 0
    #             , np.log(self.I.iloc[:, col] / self.I.iloc[:, col + 1])
    #             )

    # def _estimate_sigma2_hat(self):
    #     """
    #     Estimate the sigma2.hat values for the log ratios of the claims payments triangle. The sigma2.hat values
    #     are the variance of the log ratios for each development year. This method calculates the unbiased estimate
    #     of the variance for each development year using the log ratios of the claims payments triangle.
    #     """
    #     # for all years but the last, sigma2.hat is the variance of the self.logP values for that year
    #     self.sigma2_hat = pd.Series(np.var(self.logP.iloc[:, :-1], axis=0), index=self.logP.columns[:-1])

    #     # for the last year, sigma2.hat is estimated by log-linear regression on the values of self.sigma2_hat
    #     # already calculated
    #     self.sigma2_hat[self.logI.columns[self.logI.shape[1] -1]] = self._estimate_sigma2_hat_last_year()[0]

    #     # reset the index
    #     self.sigma2_hat = self.sigma2_hat.reset_index(drop=True)

    # def _estimate_sigma2_hat_last_year(self):
    #     """
    #     Estimate the sigma2.hat value for the last development year using log-linear regression on the values of
    #     sigma2.hat already calculated for the previous development years.
    #     """
    #     y = pd.Series(np.log(np.var(self.logP.iloc[:, :-1], axis=0)), index=self.logP.columns[:-1])
    #     X = pd.Series(np.arange(1, len(y) + 1), index=y.index)

    #     # fit a ridge regression model to the data
    #     ridge = Ridge(alpha=1.0, fit_intercept=True)
    #     ridge.fit(X.values.reshape(-1, 1), y.values)

    #     # return the predicted value for the last development year
    #     return np.exp(ridge.predict(np.array([len(y)]).reshape(-1, 1)))

    # def _estimate_tau2_hat(self):
    #     """
    #     Estimate the tau2.hat values for the log ratios of the incurred losses triangle. The tau2.hat values are
    #     the variance of the log ratios for each development year. This method calculates the unbiased estimate of
    #     the variance for each development year using the log ratios of the incurred losses triangle.
    #     """
    #     # for all years but the last two, tau2.hat is the variance of the self.logI values for that year
    #     self.tau2_hat = pd.Series(np.var(self.logI.iloc[:, :-2], axis=0), index=self.logI.columns[:-2])

    #     # for the last two years, tau2.hat is estimated by log-linear regression on the values of self.tau2_hat
    #     # already calculated
    #     self.tau2_hat[self.logI.columns[-1]] = self._estimate_tau2_hat_last_year()

    #     # reset the index
    #     self.tau2_hat = self.tau2_hat.reset_index(drop=True)

    # def _estimate_tau2_hat_last_year(self):
    #     """
    #     Estimate the tau2.hat value for the last two development years using log-linear regression on the values of
    #     tau2.hat already calculated for the previous development years.
    #     """
    #     y = pd.Series(np.log(np.var(self.logI.iloc[:, :-2], axis=0)), index=self.logI.columns[:-2])
    #     X = pd.Series(np.arange(1, len(y) + 1), index=y.index)

    #     # fit a ridge regression model to the data
    #     ridge = Ridge(alpha=1.0, fit_intercept=True)
    #     ridge.fit(X.values.reshape(-1, 1), y.values)

    #     # return the predicted value for the last development year
    #     return np.exp(ridge.predict(np.array([len(y)]).reshape(-1, 1)))

    # def _estimate_v2_hat(self):
    #     """
    #     Current R code:
    #     # v2_j estimates, j=1,...,J
    #     v2 <- rep(NA,J)
    #     for (i in 1:(J-1)) {
    #         v2[i] <- sum(sigma2.hat) + sum(tau2.hat[i:J-1])
    #     }
    #     v2[J] <- sum(sigma2.hat)
    #     """
    #     # initialize array of length J full of zeros
    #     self.v2_hat = pd.Series(np.zeros(len(self.sigma2_hat)), index=self.sigma2_hat.index)

    #     # each element of v2_hat is the sum of total sum of sigma2_hat and the sum of tau2_hat for all
    #     # development years greater than or equal to the current development year (eg i to J-1)
    #     for i in range(len(self.v2_hat)):
    #         self.v2_hat.loc[i] = np.sum(self.sigma2_hat) + np.sum(self.tau2_hat[i:])

    #     # the last element of v2_hat is just the sum of sigma2_hat
    #     self.v2_hat.loc[len(self.v2_hat) - 1] = np.sum(self.sigma2_hat)

    #     # reset the index
    #     self.v2_hat = self.v2_hat.reset_index(drop=True)

    # def _estimate_w2_hat(self):
    #     """
    #     Current R code:
    #     w2 <- rep(NA,J)
    #     for (i in 1:J) {
    #         w2[i] <- sum(sigma2.hat[1:i])
    #     }
    #     """
    #     # initialize array of length J full of zeros
    #     self.w2_hat = pd.Series(np.zeros(self.J), index=self.sigma2_hat.index)

    #     # each element of w2_hat is the sum of sigma2_hat for all development years less than or equal to the
    #     # current development year (eg 1 to i)
    #     for i in self.sigma2_hat.index.tolist():
    #         self.w2_hat.loc[i] = np.sum(self.sigma2_hat.loc[:i])

    # def _estimate_c_hat(self):
    #     """
    #     Calculate the vector c of log-linear parameters.

    #     The vector c contains the log-linear parameters for each development year.
    #     The first element of c is the sum of the log ratios of the incurred losses triangle divided by
    #     the paid losses triangle for the first development year.
    #     The remaining elements of c are calculated using the formula:
    #     c[j] = (1/sigma2_hat[j]) * sum(logP[:J+1-j,j]) +
    #     sum(1/(v2[:j-1]-w2[:j-1]) * log(diag(I[J-j+1:J,:j-1])/diag(P[J-j+1:J,:j-1])))

    #     Returns:
    #     None

    #     Current R code:
    #     c <- c()
    #     for (j in 1:J)	{
    #         if (j==1) {
    #         c[j] <- (1/sigma2.hat[j]) * sum(fP[1:J,j])
    #         }
    #         else if (j==2) {
    #         c[j] <- (1/sigma2.hat[j]) * sum(fP[1:(J+1-j),j]) +
    #         sum( 1/(v2[1:(j-1)]-w2[1:(j-1)]) *
    #         log(triangleI[J:(J-j+2),1:(j-1)]/triangleP[J:(J-j+2),1:(j-1)]))
    #         } else {
    #             diag2 <- diag(triangleI[J:(J-j+2),1:(j-1)])
    #             diag <- diag(triangleP[J:(J-j+2),1:(j-1)])
    #             c[j] <- (1/sigma2.hat[j]) * sum(fP[1:(J+1-j),j]) +
    #             sum( 1/(v2[1:(j-1)]-w2[1:(j-1)]) *
    #             log(diag2[1:(j-1)]/diag[1:(j-1)]))
    #         }
    #     }
    #     """
    #     # Initialize an array of length J full of zeros
    #     c = np.zeros(self.J)

    #     # The first element of c
    #     c[0] = (1 / self.sigma2_hat[0]) * np.sum(self.logP.iloc[:, 0])

    #     print(f"J: {self.J} \ shape of self.P: {self.P.shape} \ shape of self.I: {self.I.shape}\n")

    #     # print(f"sigma2_hat:\n{self.sigma2_hat}\n")
    #     # print(f"w2_hat:\n{self.w2_hat}\n")
    #     # print(f"v2_hat:\n{self.v2_hat}\n\n")

    #     # Loop over each development year from 1 to J-1
    #     for j in range(1, self.J):
    #         print(f"j, row index, col index: {j}, {self.J-j}, {j}\n")

    #         # Get the submatrices of I and P needed for the calculations
    #         tri_I = self.I.iloc[self.J-j:self.J, :j]
    #         tri_P = self.P.iloc[self.J-j:self.J, :j]

    #         # tri_I = self.I.iloc[self.J-j+1:self.J, :j]
    #         # tri_P = self.P.iloc[self.J-j+1:self.J, :j]

    #         print(f"tri_I:\n{tri_I}\n")
    #         print(f"tri_P:\n{tri_P}\n\n")

    #         # this is currently getting the diagonal of the entire triangle
    #         diag_I = np.diag(np.fliplr(tri_I))
    #         diag_P = np.diag(np.fliplr(tri_P))

    #         print(f"diag_I: {diag_I}\n")
    #         print(f"diag_P: {diag_P}\n\n")

    #         print(f"(1 / self.sigma2_hat[j])={1 / self.sigma2_hat[j]}\n")
    #         print(f"np.sum(self.logP.iloc[self.J-j:self.J, j].values)=\n{np.sum(self.logP.iloc[self.J-j:self.J, j].values)}\n")
    #         print(f"1 / (self.v2_hat[:j] - self.w2_hat[:j])=\n{1 / (self.v2_hat[:j] - self.w2_hat[:j])}\n")
    #         print(f"np.log(diag_I / diag_P)=\n{np.log(diag_I / diag_P)}\n\n")

    #         if j == 1:
    #             c[j] = (1 / self.sigma2_hat[j]) * np.sum(self.logP.iloc[self.J-j:self.J, j].values) \
    #                     + np.sum(1 / (self.v2_hat[:j] - self.w2_hat[:j]) * np.log(diag_I / diag_P))

    #         else:
    #             c[j] = (1 / self.sigma2_hat[j]) * np.sum(self.logP.iloc[self.J-j:self.J, j].values) \
    #                     + np.sum(1 / (self.v2_hat[:j-1] - self.w2_hat[:j-1]) * np.log(diag_I / diag_P))

    #     self.c = c

    #     # # Initialize an array of length J full of zeros
    #     # c = np.zeros(self.J)

    #     # # The first element of c is the sum of the log ratios of the incurred losses triangle divided by
    #     # # the paid losses triangle for the first development year
    #     # # We can calculate this separately since it requires a different slice of self.logP and only depends on sigma2_hat[0]
    #     # c[0] = np.sum(self.logI.iloc[:, 0] / self.P.iloc[:, 0]) / self.M

    #     # print(f"self.J = {self.J}")
    #     # print(f"self.M = {self.M}")
    #     # print(f"self.sigma2_hat = {self.sigma2_hat}")

    #     # # Loop over each development year from 2 to J-1
    #     # for j in range(2, self.J):
    #     #     print(f"j = {j}")
    #     #     print(f"row: {self.J-j+1}:{self.J}")
    #     #     print(f"col: {0}:{j-1}")
    #     #     # Get the submatrices of I and P needed for the calculations
    #     #     tri_I = self.I.iloc[self.J-j:self.J-1, :j-1]
    #     #     tri_P = self.P.iloc[self.J-j:self.J-1, :j-1]

    #     #     print(f"tri_I = {tri_I}")
    #     #     print(f"tri_P = {tri_P}")

    #     #     # Extract the diagonal elements of the submatrices using np.diag()
    #     #     diag_I = np.diag(tri_I)
    #     #     diag_P = np.diag(tri_P)

    #     #     print(f"diag_I: {diag_I}")
    #     #     print(f"diag_P: {diag_P}")

    #     #     # Calculate the element of c using vectorized operations
    #     #     # First, calculate the sum of the loiloc ratios of the incurred losses triangle divided by
    #     #     # the paid losses triangle for the current development year
    #     #     # Then, calculate the sum of the log ratios of the incurred losses triangle divided by
    #     #     # the paid losses triangle for the previous development years using vectorized operations
    #     #     # Finally, calculate the sum of the log ratios of the diagonal elements of I divided by the diagonal
    #     #     # elements of P for the previous development years using vectorized operations

    #     #     # tempA = self.logP.iloc[self.J-j:self.J, j-1]
    #     #     tempA = (self.P.iloc[self.J-j, j-1] + 1e-9) / (self.sigma2_hat[j-1] + 1e-9)

    #     #     print(f"tempA: {tempA}")
    #     #     tempB = self.sigma2_hat[j-1]
    #     #     print(f"tempB: {tempB}")

    #     #     temp1 = np.sum(tempA / tempB)
    #     #     print(f"temp1: {temp1}")
    #     #     print(f"self.v2_hat[:j-1] = {self.v2_hat[:j-1]}\nsize: {self.v2_hat[:j-1].size}")
    #     #     print(f"self.w2_hat[:j] = {self.w2_hat[:j]}\nsize: {self.w2_hat[:j].size}")
    #     #     # print(f"temp_arr1 = self.v2_hat[:j-1] - self.w2_hat[:j] = {self.v2_hat[:j-1]} - {self.w2_hat[:j]}")
    #     #     temp_arr1 = (self.v2_hat[:j-1] - self.w2_hat[:j])
    #     #     print(f"temp_arr1: {temp_arr1}")
    #     #     print(f"temp_arr2 = np.log(diag_I / diag_P) = np.log({diag_I} / {diag_P})")
    #     #     temp_arr2 = np.log(np.divide(diag_I, diag_P))
    #     #     print(f"temp_arr2: {temp_arr2}")
    #     #     print(f"temp_arr3 = temp_arr2 / temp_arr1 = {temp_arr2} / {temp_arr1.values}")
    #     #     temp_arr3 = np.divide(temp_arr2, temp_arr1)
    #     #     print(f"temp_arr3: {temp_arr3}")
    #     #     temp2 = np.sum(temp_arr3)
    #     #     print(f"temp2: {temp2}")
    #     #     c[j-1] = (np.sum(self.logP.iloc[self.J-j:self.J, j-1] / self.sigma2_hat[j-1])
    #     #               + np.sum(1 / (self.v2_hat[:j-1] - self.w2_hat[:j]) * np.log(diag_I / diag_P)))

    #     # # The last element of c is the sum of the log ratios of the incurred losses triangle divided by
    #     # # the paid losses triangle for the final development year
    #     # c[-1] = np.sum(self.logP.iloc[:, -1] / self.sigma2_hat[-1])

    #     # # Set self.c to the resulting array of c values
    #     # self.c = c

    # def _estimate_b_hat(self):
    #     """
    #     Calculate the vector b of log-linear parameters.

    #     The vector b contains the log-linear parameters for each incremental payment period.
    #     The first element of b is calculated using a different formula than the rest of the elements.
    #     The remaining elements of b are calculated using the formula:
    #     b[j] = -(1/tau2_hat[j]) * sum(logI[:J-j,j]) -
    #     sum(1/(v2[:j]-w2[:j]) * log(diag(I[J-j:J-1,:j])/diag(P[J-j:J-1,:j])))

    #     Returns:
    #     None

    #     Current R code:
    #     b <- c()
    #     for (j in 1:(J-1))  {
    #         if (j==1) {
    #             b[j] <- -(1/tau2.hat[j]) * sum(fI[1:(J-j),j]) -
    #             sum( 1/(v2[1:j]-w2[1:j]) *
    #             log(triangleI[J:(J-j+1),1:j]/triangleP[J:(J-j+1),1:j]))
    #         } else {
    #             diag2 <- diag(triangleI[J:(J-j+1),1:j])
    #             diag <- diag(triangleP[J:(J-j+1),1:j])
    #             b[j] <- -(1/tau2.hat[j]) * sum(fI[1:(J-j),j]) -
    #             sum( 1/(v2[1:j]-w2[1:j]) *
    #             log(diag2[1:j]/diag[1:j]))
    #         }
    #     }
    #     """
    #     # Initialize an array of length J-1 full of zeros
    #     b = np.zeros(self.J - 1)

    #     # Loop over each development year from 1 to J-1
    #     for j in range(1, self.J - 1):
    #         # Calculate the element of b using vectorized operations
    #         if j == 1:
    #             # For the first development year, we use a different slice of logI and tau2_hat
    #             b[j-1] = -(1/self.tau2_hat[j-1]) * np.sum(self.logI[:self.J-j, j-1]) \
    #                       - np.sum(1 / (self.v2_hat[:j] - self.w2_hat[:j]) *
    #                                np.log(self.I[self.J-j:self.J-1, :j] / self.P[self.J-j:self.J-1, :j]))
    #         else:
    #             # For the remaining development years, we extract the diagonal elements using np.diag()
    #             # and use vectorized operations to calculate the element of b
    #             diag_I = np.diag(self.I[self.J-j:self.J-1, :j])
    #             diag_P = np.diag(self.P[self.J-j:self.J-1, :j])
    #             b[j-1] = -(1/self.tau2_hat[j-1]) * np.sum(self.logI[:self.J-j, j-1]) \
    #                       - np.sum(1 / (self.v2_hat[:j] - self.w2_hat[:j]) * np.log(diag_I[:j] / diag_P[:j]))

    #     # Set self.b to the resulting array of b values
    #     self.b = b

    # def _estimate_Ainv(self):
    #     """
    #     Calculate the inverse of the covariance matrix A.
    #     """
    #     J = self.J
    #     sigma2_hat = self.sigma2_hat
    #     tau2_hat = self.tau2_hat
    #     v2_hat = self.v2_hat
    #     w2_hat = self.w2_hat

    #     # Create indices for the diagonal and off-diagonal elements.
    #     # diag_indices will be a tuple containing two arrays of equal length,
    #     # each representing the row and column indices of the diagonal elements.
    #     diag_indices = np.diag_indices(2 * J - 1)

    #     # off_diag_indices will be a tuple containing two arrays of equal length,
    #     # each representing the row and column indices of the off-diagonal elements (upper right block).
    #     off_diag_indices = np.diag_indices_from(diag_indices, offset=J)

    #     # Initialize A as a zero-filled (2 * J - 1) x (2 * J - 1) matrix.
    #     A = np.zeros((2 * J - 1, 2 * J - 1))

    #     # Set the diagonal elements of A.
    #     # We concatenate two arrays: one for the elements corresponding to sigma2_hat, and one for tau2_hat.
    #     # The first array is created by dividing an array of the form [J, J-1, ..., 1] by sigma2_hat.
    #     # The second array is created by dividing an array of the form [J-1, J-2, ..., 1] by tau2_hat.
    #     # After concatenation, we add the sum of (1 / (v2_hat - w2_hat)) to each element of the resulting array.
    #     A[diag_indices] = np.concatenate([
    #         (J - np.arange(J)) / sigma2_hat,
    #         ((J - 1) - np.arange(J - 1)) / tau2_hat
    #     ]) + np.sum(1 / (v2_hat[:, None] - w2_hat), axis=0)

    #     # Set the off-diagonal elements of A (upper right block).
    #     # We calculate the sum of (1 / (v2_hat - w2_hat)), excluding the last element, and assign it to the off-diagonal elements.
    #     A[off_diag_indices] = -np.sum(1 / (v2_hat[:, None] - w2_hat), axis=0)[:-1]

    #     # Set the lower left block of A by symmetry.
    #     # The .T attribute transposes the matrix, so A.T[off_diag_indices] refers to the lower left block.
    #     A.T[off_diag_indices] = A[off_diag_indices]

    #     # Calculate the inverse of A to get the posterior covariance matrix.
    #     # np.linalg.inv computes the inverse of a matrix.
    #     self.Ainv = np.linalg.inv(A)

    # def _calculate_theta_posterior(self):
    #     # Combine c and b vectors
    #     cb = np.concatenate([self.c, self.b])

    #     # Calculate posterior parameters
    #     self.theta_posterior = np.dot(self.Ainv, cb)

    # def _calculate_beta_and_s2_posterior(self):
    #     J = self.J

    #     # Calculate beta
    #     self.beta = [(self.v2_hat[J] - self.w2_hat[i]) / (self.v2_hat[i] - self.w2_hat[i]) for i in range(J - 1)]

    #     # Calculate s2.posterior
    #     self.s2_posterior = np.zeros(J - 1)
    #     E = np.empty((J - 1, 2 * J - 1))

    #     for i in range(1, J):
    #         e = np.zeros(2 * J - 1)
    #         e[J + 1 - i - 1: J - 1] = 1 - self.beta[J - i]
    #         e[2 * J - i: 2 * J - 1] = self.beta[J - i]
    #         E[i - 1, :] = e
    #         self.s2_posterior[i - 1] = e @ self.Ainv @ e.T

    # def _calculate_ultimate_loss_vector(self):
    #     J = self.J
    #     PIC_Ult = np.zeros(J - 1)

    #     for i in range(1, J):
    #         # Extract the corresponding beta value
    #         beta_i = self.beta[J - i]

    #         # Calculate the intermediate sums of theta_posterior values
    #         sum_theta1 = np.sum(self.theta_posterior[J - i + 1:J])
    #         sum_theta2 = np.sum(self.theta_posterior[2 * J - i: 2 * J - 1])

    #         # Calculate the paid-to-incurred ratio (PIR) component
    #         PIR = self.P.iloc[i, J - i] ** (1 - beta_i) * self.I.iloc[i, J - i] ** beta_i

    #         # Calculate the exponential term involving the theta_posterior values
    #         exp_theta_term = np.exp((1 - beta_i) * sum_theta1 + beta_i * sum_theta2)

    #         # Calculate the exponential term involving the v2_hat and w2_hat values
    #         exp_v2_w2_term = np.exp((1 - beta_i) * (self.v2_hat[J] - self.w2_hat[J - i]) / 2 + self.s2_posterior[i - 1] / 2)

    #         # Combine the components to compute the ultimate loss estimate for the current development year
    #         PIC_Ult[i - 1] = PIR * exp_theta_term * exp_v2_w2_term

    #     self.PIC_Ult = PIC_Ult

    # def _calculate_claims_reserves(self):
    #     # Calculate the claims reserves by subtracting the diagonal values of the paid losses triangle from the ultimate loss vector
    #     diag_P = np.diag(self.P)
    #     PIC_Ris = self.PIC_Ult - diag_P

    #     # Calculate the total claims reserves
    #     PIC_RisTot = np.sum(PIC_Ris)

    #     # Store the results as class attributes
    #     self.PIC_Ris = PIC_Ris
    #     self.PIC_RisTot = PIC_RisTot

    # def _calculate_prediction_uncertainty(self):
    #     # Initialize the mean squared error of prediction (MSEP)
    #     msep = 0

    #     # Loop over i and k from the second development year to the last development year
    #     for i in range(1, self.J):
    #         for k in range(1, self.J):
    #             if i == k:
    #                 # Update MSEP for the case when i equals k
    #                 msep += ((np.exp((1 - self.beta[self.J - i]) * (self.v2_hat[-1] - self.w2_hat[-1 - i])
    #                                 + self.E[i - 1].dot(self.Ainv).dot(self.E[k - 1])) - 1)
    #                         * self.PIC_Ult[i - 1] * self.PIC_Ult[k - 1])
    #             else:
    #                 # Update MSEP for the case when i is not equal to k
    #                 msep += ((np.exp(self.E[i - 1].dot(self.Ainv).dot(self.E[k - 1])) - 1)
    #                         * self.PIC_Ult[i - 1] * self.PIC_Ult[k - 1])

    #     # Calculate the prediction standard error (PIC.se) as the square root of the MSEP
    #     PIC_se = np.sqrt(msep)

    #     # Store the result as a class attribute
    #     self.PIC_se = PIC_se

    # def _fit_frequentist(self
    #                      , return_df : bool = True
    #                      ) -> Union[pd.DataFrame, None]:
    #     # Calculate log ratios
    #     self._calculate_log_ratios()

    #     # Estimate sigma2.hat and tau2.hat
    #     self._estimate_sigma2_hat()
    #     self._estimate_tau2_hat()

    #     # Estimate v2.hat and w2.hat
    #     self._estimate_v2_hat()
    #     self._estimate_w2_hat()

    #     # Estimate c.hat and b.hat
    #     self._estimate_c_hat()
    #     self._estimate_b_hat()

    #     # Estimate Ainv
    #     self._estimate_Ainv()

    #     # Calculate the posterior theta
    #     self._calculate_theta_posterior()

    #     # Calculate beta and s2.posterior
    #     self._calculate_beta_and_s2_posterior()

    #     # Calculate the ultimate loss vector
    #     self._calculate_ultimate_loss_vector()

    #     # Calculate the claims reserves
    #     self._calculate_claims_reserves()

    #     # Calculate the prediction uncertainty
    #     self._calculate_prediction_uncertainty()

    #     # Create the results DataFrame
    #     results_df = pd.DataFrame({
    #         'Cumulative Paid': np.diag(self.P),
    #         'Cumulative Incurred': np.diag(self.I),
    #         'Ultimate Loss': self.PIC_Ult,
    #         'Reserve': self.PIC_Ris,
    #         'Prediction Error': self.PIC_se
    #     }, index=self.P.index)
    #     self.ultimates = results_df

    #     # Return the results DataFrame if return_df is True
    #     if return_df:

    #         # Return the results DataFrame
    #         return results_df
