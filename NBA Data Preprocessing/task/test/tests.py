from hstest import StageTest, TestCase, CheckResult
from hstest.stage_test import List
import pandas as pd
import os

module = True
type_err = True
other_err = True
try:
    from preprocess import clean_data, feature_data, multicol_data, transform_data

    path = "../Data/nba2k-full.csv"
    answer = transform_data(multicol_data(feature_data(clean_data(path))))
except ImportError:
    module = False
    clean_data = None
    feature_data = None
    multicol_data = None
    transform_data = None
except TypeError as type_err_exc:
    type_err_exc_message = type_err_exc
    type_err = False
except Exception as other_exc:
    other_exc_message = other_exc
    other_err = False


class Tests(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        if not os.path.exists('preprocess.py'):
            return CheckResult.wrong('The file `preprocess.py` is not found. Your solution should be located there.\n'
                                     'Please do not rename the file.')

        if not module:
            return CheckResult.wrong('Either functions `clean_data`, `feature_data`, or `multicol_data`\n'
                                     'from the previous stages or the function `transform_data` were not found in your solution.\n'
                                     'Please include all of them.')

        if not type_err:
            return CheckResult.wrong(f"An error occurred during execution of your solution.\n"
                                     f"The function `transform_data` should take one input parameter: DataFrame returned by `multicol_data` function.\n"
                                     f"An internal error message:\n{type_err_exc_message}")
        if not other_err:
            return CheckResult.wrong(f"An error occurred during execution of `transform_data` function.\n"
                                     f"The error message:\n{other_exc_message}\n\n"
                                     f"Refer to the Objectives and Examples sections.")
        if answer is None:
            return CheckResult.wrong(
                'The `transform_data` function returns nothing while it should return X DataFrame and y series')

        if len(answer) != 2:
            return CheckResult.wrong("The transform_data function should return X and y")

        first, second = answer

        if isinstance(first, pd.DataFrame) and isinstance(second, pd.Series):
            X, y = answer
        elif isinstance(first, pd.Series) and isinstance(second, pd.DataFrame):
            return CheckResult.wrong('Return X DataFrame and y series as X,y not y, X')
        else:
            return CheckResult.wrong('Return X as a DataFrame and y as a series')

        if X.shape != (439, 46):
            return CheckResult.wrong('X DataFrame has wrong shape')

        if y.shape != (439,):
            return CheckResult.wrong('y series has wrong shape')

        try:
            sorted(list(X.columns.str.lower()[:3]))
        except TypeError:
            return CheckResult.wrong(
                "The first three column names in the X DataFrame contain both numerical and string types.")
        except AttributeError:
            return CheckResult.wrong("The first three column names in the X DataFrame should be of type 'string'")
        except Exception:
            return CheckResult.wrong("Error while parsing X DataFrame column names")

        if sorted(list(X.columns.str.lower()[:3])) != sorted(['rating', 'experience', 'bmi']):
            return CheckResult.wrong(
                f"Your set of numerical features is currently as follows: {list(X.columns.str.lower()[:3])}.\n"
                f"This set is incorrect. Check whether you concatenated transformed features in correct order.")

        try:
            sorted(list(X.columns.str.lower()[3:]))
        except TypeError:
            return CheckResult.wrong(
                "The categorical column names contain both numerical and string types.")
        except AttributeError:
            return CheckResult.wrong(
                "The categorical columns could not be converted to string. Please check your code.")
        except Exception:
            return CheckResult.wrong("Error while parsing X DataFrame column names")

        if sorted(list(X.columns[3:])) != sorted(
                ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
                 'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
                 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
                 'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
                 'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
                 'No Team', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
                 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
                 'Utah Jazz', 'Washington Wizards', 'C', 'C-F', 'F', 'F-C', 'F-G', 'G', 'G-F',
                 'Not-USA', 'USA', '0', '1', '2']):
            return CheckResult.wrong(
                "The categorical columns are incorrect. One-hot encode the following columns: team, position, country, draft round")

        scaled_ans = [3.2352194717791973, 2.7598866876192636, 1.3201454530024874]
        student_ans = X.head(1).values.tolist()[0][:3]

        for one_scale, one_student in zip(scaled_ans, student_ans):
            if not (one_scale - .2 < one_student < one_scale + 0.2):
                return CheckResult.wrong('Standard Scaler transformation is done incorrectly')

        return CheckResult.correct()


if __name__ == '__main__':
    Tests().run_tests()
