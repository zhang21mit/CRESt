from robotic_testing.common import data_analysis

configs = {
    # file path settings
    'data_dir': 'D:/Zhichu Ren/data/temp/temp',
    'recording_dir': 'D:/Zhichu Ren/recordings',

    # email settings
    'email': {
        'chu': 'crest.mit.demo@gmail.com',
        'zhen': 'zhang21mit@gmail.com',
        'weiyin': 'chenweiyin1996@gmail.com',
        'zhen': 'zhang21mit@gmail.com',
        'Daniel': 'daniel.zheng0211@gmail.com',
        'Ali': 'ali_m@mit.edu'
    },

    # ec lab settings
    'protocol': {
        '1': 'OCV',
        '2': 'CV',
        '3': 'LSV',
    },

    # sample info
    'sample_area': 1,  # cm^2

    # data analysis settings
    'data_analysis': {
        'analyzer': {
            'analyzer_class': data_analysis.AlklineOERAnalyzer,
            'kwargs': {},
            },
        'metrics': {
            'overpotential': {
                'technique_id': '3',
                'analyze_method': data_analysis.AlklineOERAnalyzer.get_overpotential
            },
            'max_i': {
                'technique_id': '3',
                'analyze_method': data_analysis.AlklineOERAnalyzer.get_max_i
            },
            'tafel_slope': {
                'technique_id': '3',
                'analyze_method': data_analysis.AlklineOERAnalyzer.get_tafel_slope,
                'kwargs': {'range': 'current', 'I_low': 10, 'I_high': 100}
            },
            'ts_fit': {
                'technique_id': '3',
                'analyze_method': data_analysis.AlklineOERAnalyzer.get_tafel_slope_fit,
                'kwargs': {'range': 'current', 'I_low': 10, 'I_high': 100}
            },
        }
    },
}
