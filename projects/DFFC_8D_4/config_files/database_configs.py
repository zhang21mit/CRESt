from sqlalchemy import Integer, String, Float, TIMESTAMP
from sqlalchemy.sql import func

configs = {

    # schema settings
    'schema_list': [
        'active_learning',
        'botorch',
        'test_raw_data',
    ],

    # sql table settings
    'tables_dict': {
        'active_learning.recipe': {
            'schema': 'active_learning',
            'table_name': 'recipe',
            'columns_dict': {
                'arm_name': {'type_': String(50), 'primary_key': True},
                'trial_index': {'type_': Integer},
                'arm_index': {'type_': Integer},
                'Pd': {'type_': Float},
                'Pt': {'type_': Float},
                'Cu': {'type_': Float},
                'Au': {'type_': Float},
                'Ir': {'type_': Float},
                'Ce': {'type_': Float},
                'Nb': {'type_': Float},
                'Cr': {'type_': Float},
                'comments': {'type_': String(500)},
                'pred_mean': {'type_': Float},
                'pred_std': {'type_': Float},
                'created_at': {'type_': TIMESTAMP, 'server_default': func.now()},
                'updated_at': {'type_': TIMESTAMP, 'server_default': func.now(), 'onupdate': func.now()}
            }
        },

        'active_learning.sample': {
            'schema': 'active_learning',
            'table_name': 'sample',
            'columns_dict': {
                'id': {'type_': Integer, 'primary_key': True, 'autoincrement': True},
                'trial_index': {'type_': Integer},
                'sample_batch_id': {'type_': Integer},
                'arm_name': {'type_': String(50)},
                'comments': {'type_': String(500)},
                'created_at': {'type_': TIMESTAMP, 'server_default': func.now()},
                'updated_at': {'type_': TIMESTAMP, 'server_default': func.now(), 'onupdate': func.now()}
            }
        },

        'active_learning.performance': {
            'schema': 'active_learning',
            'table_name': 'performance',
            'columns_dict': {
                'id': {'type_': Integer, 'primary_key': True, 'autoincrement': True},
                'sample_id': {'type_': Integer},
                'max_power': {'type_': Float},
                'comments': {'type_': String(500)},
                'abandoned': {'type_': Integer},
                'abandon_reason': {'type_': String(50)},
                'outlier': {'type_': Integer},
                'created_at': {'type_': TIMESTAMP, 'server_default': func.now()},
                'updated_at': {'type_': TIMESTAMP, 'server_default': func.now(), 'onupdate': func.now()}
            }
        },
    },

}
