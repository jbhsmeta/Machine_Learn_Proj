import pandas as pd
import matplotlib.pyplot as plt

class df_summary():

    def __init__(self,df,target,prob_type="classi"):
        self.df=df
        self.target=target
        self.df[self.target] = self.df[self.target].astype('str')
        self.prob_type=prob_type
        self.mul_factor=0.25
        self.df_summary=self.summary()


    def null_cls_distribution(self):

        from collections import defaultdict
        self.cls_dict = defaultdict(int)
        self.match_idx = defaultdict(list)

        cls_lst = self.df[self.target].value_counts().index.tolist()
        self.cls_idx_dict = {cls:self.df[self.df[self.target] == cls].index.tolist() for cls in cls_lst}

        for col in self.df.columns:
            null_idxs_in_col = set(self.df[self.df[col].isna()].index.tolist())

            for key in self.cls_idx_dict.keys():
                no_of_idxs_in_cls = set(self.cls_idx_dict[key])
                len_idx_in_cls = len(no_of_idxs_in_cls)
                len_intersect_cls_col_idx = len(null_idxs_in_col.intersection(no_of_idxs_in_cls))
                pct_null_contri_in_cls = round((len_intersect_cls_col_idx / len_idx_in_cls) * 100, 1)
                self.match_idx[key].append(str(len_intersect_cls_col_idx) + " - " + str(pct_null_contri_in_cls) + "%")

        col_names = [cls + "_p_NL" for cls in self.cls_idx_dict.keys()]
        temp_df = pd.DataFrame(self.match_idx)
        temp_df.columns = col_names
        return temp_df


    def cls_imbalance_check(self):

        # mul_factor - Each class should have a quarter of equally distributed observation to be a balanced dataset
        df_cls_dist = self.df[self.target].value_counts(normalize=True).mul(100).round(3).reset_index(name="dist")

        # Calculation of the threshold for verification of imbalance in dataset
        imb_check_pct = (100 / df_cls_dist.shape[0]) * self.mul_factor

        if df_cls_dist.dist.any() < imb_check_pct:
            imb_dec = "Imbalanced dataset"
        else:
            imb_dec = "Balanced dataset"

        return (df_cls_dist, imb_dec)

    def get_isimbalanced(self):

        cls_imb_data, is_imb = self.cls_imbalance_check()
        print("This is ", is_imb, "\n")
        print(cls_imb_data)

    def get_summary(self):

        return(self.df_summary)

    def summary(self):

        summ_df = self.df.isnull().sum().reset_index(name="Null")
        summ_df["dtype"] = self.df.dtypes.values
        summ_df['unique'] = [len(self.df[col].value_counts().index.tolist()) for col in self.df.columns]
        summ_df["pct_Null"] = (summ_df.Null / self.df.shape[0]).mul(100).round(2).astype("str") + "%"

        df_null_dist = self.null_cls_distribution()

        df_final = pd.concat([summ_df, df_null_dist], axis=1)
        df_final = df_final.set_index("index")

        return df_final

    def print_feat(self,feature):
        print("1st 10 data in {}".format(feature))
        print(self.df[feature][:10], '\n')
        print(self.df[feature].describe(), '\n')
        print(self.df[feature].value_counts(normalize=True), '\n')
        print(self.df_summary.loc[feature])
        dtype_list = ["int64", "float64"]
        if self.df[feature].dtypes in dtype_list:
            plt.figure(figsize=(8, 5))
            self.df[feature].plot(kind="hist", bins=12)
            plt.show()
        else:
            self.df[feature].value_counts(normalize=True)[:10].plot.barh()
            plt.show()

    def univariate_ana(self, features="all"):
       '''
       :param features:
       features - can be a list or string
       if string - "all" will print uni-variate analysis of all features or single feature name can be provided as input
       if list - uni-variate analysis of the list of features will be printed
       :return:

       print the a report
       1. 1st 10 content of the feature
       2. description of the feature - using pandas describe
       3. value counts
       4. slice from summary df showing null values, dtype, uniques values, pct of null values,distribution of null
       values in classes
       '''
       import os
       c=1
       if isinstance(features, str):
           if features == "all":
               for feat in self.df.columns:
                   if c%5==0: # after printing first 5, it will ask permission to go ahead
                       os.system("pause")
                       inp = input("Please press y/Y to continue n/N to exit -  ")
                       if inp.lower() == "y":
                           continue
                       else:
                           break
                   else:
                       self.print_feat(feat)


           elif features in self.df.columns:
               self.print_feat(features)

           else:
               print("Feature not in the dataframe. Please Check!!!")

       elif isinstance(features,list):
           for feat in features:
               if c % 5 == 0:  # after printing first 5, it will ask permission to go ahead
                   os.system("pause")
                   inp = input("Please press y/Y to continue n/N to exit -  ")
                   if inp.lower() == "y":
                       continue
                   else:
                       break
               else:
                   self.print_feat(feat)


    def bivariate_ana(self,var_x, var_y, cr_tab=True, plot=False):
        df_bi_vt = pd.crosstab(self.df[var_x], self.df[var_y])

        if not plot:
            if cr_tab:
                return df_bi_vt
        else:
            pd.crosstab(self.df[var_x], self.df[var_y]).reset_index().plot(x=var_x, kind='bar', stacked=True,
                                                                 title='{} vs {}'.format(var_x, var_y))
            return df_bi_vt
