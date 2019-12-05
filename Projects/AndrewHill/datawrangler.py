import numpy as np
import functools
import h5py
import pandas as pd
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

SAMPLE_IDX = 31128

class Normalizer(object):
    """
    Normalizer class allows you to layer normalization modules on top of each other in sequence.
    """
    def __init__(self, layers=None):
        self.layers = layers if layers is not None else []
        self.padding = 5
        self.samples = []
    def apply(self,data):
        # Formatting for output table.
        col_width = np.max([len(type(l).__name__) for l in self.layers]) + self.padding
        in_width = 20
        out_width = 20
        header = "|".join(["Module".center(col_width),"Input Shape".center(in_width),"Output Shape".center(out_width)])
        header_bar = "-"*len(header)
        print(header)
        print(header_bar)
        # Apply each layer in sequence.
        for layer in self.layers:
            data,sample_tup = layer.apply(data,col_width=col_width,in_width=in_width,out_width=out_width)
            self.samples.append(sample_tup)
        print(header_bar)
        # Return the output of the final normalization layer.
        return data

    def plot_samples(self):
        # Plot each sample along with a descriptive title.
        # Also save them to file.
        for idx,(smpl,mt_name) in enumerate(self.samples):
            plt.figure(figsize=(9,7))
            plt.plot(smpl.values)
            plt.title("Subject %d after Normalization Step %d (%s)" % (SAMPLE_IDX,idx,mt_name))
            plt.savefig("Step_%d.PNG" % idx)
            plt.close()


class Method(object):
    """
    Base class for a Method.

    Just use this so that we can derive from it and use this decorator.
    """

    # Decorator wraps around normalization methods to provide the input/output table and
    # the sample generation.
    @classmethod
    def normalization_method(self, func):
        def inner(self,data,col_width,in_width,out_width,*args,**kwargs):
            # Get the shape of the data before the normalization step
            input_shape = str(data.shape).center(in_width)
            # Get the class name of the module that is being applied
            method_name = type(self).__name__.ljust(col_width)
            # Apply the normalization method
            data = func(self,data,*args,**kwargs)
            # Get the shape of the data after normalization
            output_shape = str(data.shape).center(out_width)
            # Make a formatted string containing all of this information
            method_str = "|".join([method_name,input_shape,output_shape])
            if input_shape != output_shape:
                method_str = '\033[1m' + method_str + '\033[0m'
            # Print to console
            print(method_str)
            # Retrieve a data sample.
            sample = data.loc[SAMPLE_IDX,:]
            # Return the data, and a tuple of (Sample data, Method Name) for displaying samples.
            return data,(sample,method_name.replace(" ",""))
        return inner


class ZTransformNormalize(Method):
    """
    ZTransformNormalize

    Transform data measurements into Z-scores from a standard Normal distribution.
    """
    def __init__(self,axis):
        self.axis = axis
        super().__init__()

    @Method.normalization_method
    def apply(self,data):
        # Subtract the mean of the data along an axis and then divide by the standard deviation on the same axis.
        # Transforms the data into a Z-score from the standard normal distribution.
        data = pd.DataFrame((data.values - data.values.mean(axis=self.axis,keepdims=True))/data.values.std(axis=self.axis,keepdims=True),columns=data.columns,index=data.index)
        return data

class MinMaxNormalize(Method):
    """
    MinMaxNormalize

    Normalize the data to fall between 0 and 1 based on min and max values for a given axis.
    """
    def __init__(self,axis):
        self.axis = axis

    @Method.normalization_method
    def apply(self,data):
        # Calculate minimum and maximum value along the axis, then normalize each value to be between 0-1 based on the range.
        data_mins = np.min(data.values,axis=self.axis,keepdims=True)
        data_maxs = np.max(data.values,axis=self.axis,keepdims=True)
        data = pd.DataFrame((data.values - data_mins)/(data_maxs - data_mins),columns=data.columns,index=data.index)
        return data

class NaNReplacer(Method):
    """
    NaNReplacer

    Replace all NaN values with a specified constant value.
    """
    def __init__(self,const_val=0):
        self.const_val = const_val
    @Method.normalization_method
    def apply(self,data):
        # Find all NaN values and replace them with a constant value.
        nans = np.isnan(data)
        data[nans] = self.const_val
        return data

class ConstValueDropper(Method):
    """
    ConstValueDropper

    Simple method to drop rows/columns that contain all constant value.
    """
    def __init__(self,axis):
        self.axis = axis

    @Method.normalization_method
    def apply(self,data):
        # If all values along the given axis have the same value, then remove that row/column.
        is_const_val = np.all(np.equal(data.values,np.min(data.values,axis=self.axis,keepdims=True)),axis=self.axis)
        data = data[~is_const_val]
        return data

class LongToWideFormat(Method):
    """
    LongToWideFormat

    Convert a "Long" format with one measurement per row to a "Wide" format data with multiple measurements per row.
    Wrapper around Pandas "pivot" function
    """
    def __init__(self,index_col,data_col,timestamp_col):
        self.index_col = index_col
        self.data_col = data_col
        self.timestamp_col = timestamp_col

    @Method.normalization_method
    def apply(self,data):
        return data.pivot(index=self.index_col,values=self.data_col,columns=self.timestamp_col)

class TimeseriesResampler(Method):
    """
    TimeseriesResampler

    Wrapper around pandas resample function to perform time series resampling based on a time-based Index.
    """
    def __init__(self,args,**kwargs):
        self.args = args
        self.kwargs = kwargs

    @Method.normalization_method
    def apply(self,data):
        return data.resample(self.args,**self.kwargs).sum()

class StableSeasonalFilter(Method):
    """
    StableSeasonalFilter

    Simple version of a Stable Seasonal Filter to remove seasonality trends from the data.
    """
    def __init__(self,num_seasons):
        self.num_seasons = num_seasons

    @Method.normalization_method
    def apply(self,data):
        tmp = data.values
        len_season = tmp.shape[1]//self.num_seasons

        assert (tmp.shape[1] == len_season * self.num_seasons)

        conv_filt = np.full(len_season + 1, 1 / len_season)
        conv_filt[0] = 1 / (len_season * 2)
        conv_filt[-1] = 1 / (len_season * 2)
        conv_filt = np.repeat(conv_filt, tmp.shape[0]).reshape(-1, tmp.shape[0]).transpose()
        moving_avg = fftconvolve(tmp, conv_filt, "same", axes=-1)

        detrended = tmp - moving_avg

        mean_idcs = np.repeat(np.arange(len_season), self.num_seasons).reshape(-1, self.num_seasons) + (
                    np.arange(self.num_seasons) * len_season)

        mean_comps = np.expand_dims(detrended[:, mean_idcs].mean(axis=-1), axis=0)

        stab_seas_comp = np.repeat(mean_comps, self.num_seasons, axis=0).transpose(1, 0, 2).reshape(tmp.shape[0], -1)
        filtered = tmp - stab_seas_comp
        return pd.DataFrame(filtered,index=data.index,columns=data.columns)

if __name__ == "__main__":

    data = pd.read_csv("C:\\Users\\96ahi\\Documents\\DataManipulate\\PAXRAW_D\\PAXRAW_D.csv").iloc[:74874095]

    # Set t0 = 0 by subtracting the minimum value from the entire column
    data.PAXN -= data.PAXN.min()
    # Add a timedelta column to the data, so that we can resample.
    data["delta_t"] = data.groupby(by="SEQN").apply(lambda grp: pd.to_timedelta(grp.PAXN, "m")).reset_index(drop=True)

    data.to_pickle("PAXRAW_D.pkl")

    normalizer = Normalizer([LongToWideFormat(index_col="SEQN",data_col="PAXINTEN",timestamp_col="delta_t"),
                             TimeseriesResampler("30T",axis=1),
                             NaNReplacer(const_val=0),
                             ConstValueDropper(axis=1),
                             StableSeasonalFilter(num_seasons=7),
                             ZTransformNormalize(axis=-1)
                             ])
    data_df = normalizer.apply(data)

    with h5py.File("new_depict_data.h5") as fl:
        fl["data"] = np.expand_dims(np.expand_dims(data_df.values,axis=1),axis=1)[:7400]
        fl["uids"] = np.array(data_df.index).astype(np.int32)[:7400]
        fl["labels"] = np.random.choice(2,7400).astype(np.int32)
    fl.close()
    print("done!")


