import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from dpipe.dataset.wrappers import apply
from ood.dataset.cc359 import CC359
from ood.dataset.utils import Rescale3D, scale_mri
from ood.paths import CC359_DATA_PATH


train_ids = [
"CC0184",
"CC0185",
"CC0188",
"CC0193",
"CC0194",
"CC0195",
"CC0196",
"CC0197",
"CC0201",
"CC0202",
"CC0203",
"CC0207",
"CC0209",
"CC0213",
"CC0214",
"CC0215",
"CC0219",
"CC0222",
"CC0223",
"CC0224",
"CC0225",
"CC0227",
"CC0229",
"CC0230",
"CC0231",
"CC0233",
"CC0234",
"CC0236",
"CC0237",
"CC0238"
] 

valid_ids = ["CC0359"]

test_ids = [
"CC0180",
"CC0181",
"CC0182",
"CC0183",
"CC0186",
"CC0187",
"CC0189",
"CC0190",
"CC0191",
"CC0192",
"CC0198",
"CC0199",
"CC0200",
"CC0204",
"CC0205",
"CC0206",
"CC0208",
"CC0210",
"CC0211",
"CC0212",
"CC0216",
"CC0217",
"CC0218",
"CC0220",
"CC0221",
"CC0226",
"CC0228",
"CC0232",
"CC0235",
"CC0239",
"CC0001",
"CC0002",
"CC0003",
"CC0004",
"CC0005",
"CC0006",
"CC0007",
"CC0008",
"CC0009",
"CC0010",
"CC0011",
"CC0012",
"CC0013",
"CC0014",
"CC0015",
"CC0016",
"CC0017",
"CC0018",
"CC0019",
"CC0020",
"CC0021",
"CC0022",
"CC0023",
"CC0024",
"CC0025",
"CC0026",
"CC0027",
"CC0028",
"CC0029",
"CC0030",
"CC0031",
"CC0032",
"CC0033",
"CC0034",
"CC0035",
"CC0036",
"CC0037",
"CC0038",
"CC0039",
"CC0040",
"CC0041",
"CC0042",
"CC0043",
"CC0044",
"CC0045",
"CC0046",
"CC0047",
"CC0048",
"CC0049",
"CC0050",
"CC0051",
"CC0052",
"CC0053",
"CC0054",
"CC0055",
"CC0056",
"CC0057",
"CC0058",
"CC0059",
"CC0060",
"CC0061",
"CC0062",
"CC0063",
"CC0064",
"CC0065",
"CC0066",
"CC0067",
"CC0068",
"CC0069",
"CC0070",
"CC0071",
"CC0072",
"CC0073",
"CC0074",
"CC0075",
"CC0076",
"CC0077",
"CC0078",
"CC0079",
"CC0080",
"CC0081",
"CC0082",
"CC0083",
"CC0084",
"CC0085",
"CC0086",
"CC0087",
"CC0088",
"CC0089",
"CC0090",
"CC0091",
"CC0092",
"CC0093",
"CC0094",
"CC0095",
"CC0096",
"CC0097",
"CC0098",
"CC0099",
"CC0100",
"CC0101",
"CC0102",
"CC0103",
"CC0104",
"CC0105",
"CC0106",
"CC0107",
"CC0108",
"CC0109",
"CC0110",
"CC0111",
"CC0112",
"CC0113",
"CC0114",
"CC0115",
"CC0116",
"CC0117",
"CC0118",
"CC0119",
"CC0120",
"CC0121",
"CC0122",
"CC0123",
"CC0124",
"CC0125",
"CC0126",
"CC0127",
"CC0128",
"CC0129",
"CC0130",
"CC0131",
"CC0132",
"CC0133",
"CC0134",
"CC0135",
"CC0136",
"CC0137",
"CC0138",
"CC0139",
"CC0140",
"CC0141",
"CC0142",
"CC0143",
"CC0144",
"CC0145",
"CC0146",
"CC0147",
"CC0148",
"CC0149",
"CC0150",
"CC0151",
"CC0152",
"CC0153",
"CC0154",
"CC0155",
"CC0156",
"CC0157",
"CC0158",
"CC0159",
"CC0160",
"CC0161",
"CC0162",
"CC0163",
"CC0164",
"CC0165",
"CC0166",
"CC0167",
"CC0168",
"CC0169",
"CC0170",
"CC0171",
"CC0172",
"CC0173",
"CC0174",
"CC0175",
"CC0176",
"CC0177",
"CC0178",
"CC0179",
"CC0240",
"CC0241",
"CC0242",
"CC0243",
"CC0244",
"CC0245",
"CC0246",
"CC0247",
"CC0248",
"CC0249",
"CC0250",
"CC0251",
"CC0252",
"CC0253",
"CC0254",
"CC0255",
"CC0256",
"CC0257",
"CC0258",
"CC0259",
"CC0260",
"CC0261",
"CC0262",
"CC0263",
"CC0264",
"CC0265",
"CC0266",
"CC0267",
"CC0268",
"CC0269",
"CC0270",
"CC0271",
"CC0272",
"CC0273",
"CC0274",
"CC0275",
"CC0276",
"CC0277",
"CC0278",
"CC0279",
"CC0280",
"CC0281",
"CC0282",
"CC0283",
"CC0284",
"CC0285",
"CC0286",
"CC0287",
"CC0288",
"CC0289",
"CC0290",
"CC0291",
"CC0292",
"CC0293",
"CC0294",
"CC0295",
"CC0296",
"CC0297",
"CC0298",
"CC0299",
"CC0300",
"CC0301",
"CC0302",
"CC0303",
"CC0304",
"CC0305",
"CC0306",
"CC0307",
"CC0308",
"CC0309",
"CC0310",
"CC0311",
"CC0312",
"CC0313",
"CC0314",
"CC0315",
"CC0316",
"CC0317",
"CC0318",
"CC0319",
"CC0320",
"CC0321",
"CC0322",
"CC0323",
"CC0324",
"CC0325",
"CC0326",
"CC0327",
"CC0328",
"CC0329",
"CC0330",
"CC0331",
"CC0332",
"CC0333",
"CC0334",
"CC0335",
"CC0336",
"CC0337",
"CC0338",
"CC0339",
"CC0340",
"CC0341",
"CC0342",
"CC0343",
"CC0344",
"CC0345",
"CC0346",
"CC0347",
"CC0348",
"CC0349",
"CC0350",
"CC0351",
"CC0352",
"CC0353",
"CC0354",
"CC0355",
"CC0356",
"CC0357",
"CC0358",
"CC0359"
]


def get_brain_mask(t1):
    brain_mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(t1) > 0).astype(np.uint8))
    brain_mask.CopyInformation(t1)
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
    return brain_mask


def z_score_normalisation(channel, brain_mask, cutoff_percentiles=(5., 95.), cutoff_below_mean=True):
    low, high = np.percentile(channel[brain_mask.astype(bool)], cutoff_percentiles)
    norm_mask = np.logical_and(brain_mask, np.logical_and(channel > low, channel < high))
    if cutoff_below_mean:
        norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))
    masked_channel = channel[norm_mask]
    normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
    return normalised_channel


def fix_segmentation_labels(seg):
    array = sitk.GetArrayFromImage(seg)
    array[array == 4] = 3
    new_seg = sitk.GetImageFromArray(array)
    new_seg.CopyInformation(seg)
    return new_seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        required=True,
                        type=str,
                        help='Path to input directory.')
    parser.add_argument('--output-dir',
                        required=True,
                        type=str,
                        help='Path to output directory.')

    parse_args, unknown = parser.parse_known_args()
    
    data_path = CC359_DATA_PATH
    voxel_spacing = (1, 0.95, 0.95)

    preprocessed_dataset = apply(Rescale3D(CC359(data_path), voxel_spacing), load_image=scale_mri)
    dataset = apply(preprocessed_dataset, load_image=np.float32)
    
    output_dataframe = pd.DataFrame()
    for id_ in tqdm(dataset.ids):
        segm = sitk.GetImageFromArray(dataset.load_segm(id_))
        seg_path = os.path.join(parse_args.output_dir, id_) + '_seg.nii.gz'
        output_dataframe.loc[id_, 'seg'] = seg_path
        os.makedirs(os.path.dirname(seg_path), exist_ok=True)
        sitk.WriteImage(segm, seg_path)

        image = sitk.GetImageFromArray(dataset.load_image(id_))
        brain_mask = get_brain_mask(image)
        output_path = os.path.join(parse_args.output_dir, id_) + f'_brain_mask.nii.gz'
        output_dataframe.loc[id_, 'sampling_mask'] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(brain_mask, output_path)

        channel_array = sitk.GetArrayFromImage(image)
        normalised_channel_array = z_score_normalisation(channel_array, sitk.GetArrayFromImage(brain_mask))
        normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
        normalised_channel.CopyInformation(image)
        output_path = os.path.join(parse_args.output_dir, id_) + f'_t1.nii.gz'
        output_dataframe.loc[id_, 't1'] = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(normalised_channel, output_path)

    output_dataframe.index.name = 'id'
    os.makedirs('stochastic_segmentation_networks/assets/cc359_data', exist_ok=True)
    train_index = output_dataframe.loc[train_ids]
    train_index.to_csv('stochastic_segmentation_networks/assets/cc359_data/data_index_train.csv')
    valid_index = output_dataframe.loc[valid_ids]
    valid_index.to_csv('stochastic_segmentation_networks/assets/cc359_data/data_index_valid.csv')
    test_index = output_dataframe.loc[test_ids]
    test_index.to_csv('stochastic_segmentation_networks/assets/cc359_data/data_index_test.csv')
