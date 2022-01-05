from sharq import sharq


sharq_model = sharq(DATASET='labour',
                    FORECAST_GRANULARITY='M',
                    IS_HIERARCHICAL=True,
                    FORECAST_HORIZON=2,
                    VERBOSE=True,
                    TRAINING_METHOD='sharq')
sharq_model.fit_and_predict()
