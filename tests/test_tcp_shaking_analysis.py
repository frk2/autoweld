"""Vector RMS and missing-feedback handling for offline TCP metrology."""
import numpy as np

from scripts.analyze_mixed_tcp import filter_valid, vector_rms


def test_vector_band_rms_and_rotation_invariance():
    t=np.arange(0,12,.02)
    x=np.c_[.3*np.sin(2*np.pi*7*t),.4*np.cos(2*np.pi*7*t),np.zeros_like(t)]
    y,keep=filter_valid(t,x,np.ones(len(t),bool),20.)
    assert abs(vector_rms(y[keep])-np.sqrt((.3**2+.4**2)/2)) < .0002
    theta=.7
    rotation=np.array([[np.cos(theta),-np.sin(theta),0],
                       [np.sin(theta),np.cos(theta),0],[0,0,1]])
    assert abs(vector_rms(y[keep]@rotation)-vector_rms(y[keep])) < 1e-12


def test_missing_feedback_splits_filter_with_one_second_margins():
    t=np.arange(0,12,.02)
    x=np.c_[np.sin(2*np.pi*7*t)]
    valid=~((t>=5)&(t<=5.2))
    band,keep=filter_valid(t,x,valid,20.)
    assert np.isnan(band[~valid]).all()
    assert not keep[(t>=4)&(t<=6.2)].any()
    assert keep[(t>=2)&(t<=3)].all()
