import numpy as np

from uq360.utils.batch_features.feature_extractor import FeatureExtractor
from uq360.algorithms.blackbox_metamodel.predictors.base.predictor_base import PerfPredictor

from uq360.utils.utils import Timer


# This class is the new wrapper class that will drive all aspects of PP and Error Bars
class PredictorDriver(object):

    def __init__(self, pp_type,
                 base_model=None,
                 metamodels_considered=None,
                 calibrator='isotonic_regression',
                 random_state=42,
                 **kwargs):
        self.timer = Timer()
        self.timer.start('init')

        assert(metamodels_considered is not None), f'metamodels_considered not specified'
        assert(isinstance(metamodels_considered, dict)), f'metamodels_considered is not a dictionary'
        for k,v in metamodels_considered.items():
            assert(v is not None and isinstance(v, list) and len(v) > 0), f'metamodels_considered {k} missing pointwise_features'

        # Organize metamodels varaible to allow adding interim content for each metamodel
        self.metamodels = {k:{'pointwise_features':v} for k,v in metamodels_considered.items()}

        # Specify the performance predictor that you would like to run
        # Instantiate perf_predictor for each metamodel
        for k,v in self.metamodels.items():
            v['perf_predictor'] = PerfPredictor.instance(pp_type, calibrator=calibrator, metamodels_considered=[k], random_state=random_state)

        self.base_model = base_model
        print("Predictor type :", pp_type)
        print("calibrator :", calibrator)
        print("metamodels considered:", metamodels_considered)

        # instantiate feature_extractor from pointwise features found in all metamodels
        pointwise_features = sorted(set([i for v in self.metamodels.values() for i in v['pointwise_features']]))
        self.feature_extractor = FeatureExtractor(pointwise_features, None)

        self.timer.stop('init')

    def _get_column_names(self, feature_names, column_names):
        names = []
        for name in feature_names:
            if name in column_names:
                names.append(name)
            else:
                names.extend([i for i in column_names if i.startswith(f'{name}_') and i[len(name) + 1:].isdecimal()])
        return names

    def fit(self, x_train, y_train, x_test, y_test, test_predicted_probabilities=None):
        self.timer.start('fit')
        self.train_labels = np.unique(y_train)

        # Get metamodel ground truth
        if self.base_model is not None:
            test_predicted_probabilities = self.base_model.predict_proba(x_test)
            predictions = self.base_model.predict(x_test)
        else:
            try:
                assert test_predicted_probabilities is not None
            except:
                raise Exception(
                    "If base model is not provided to constructor, confidence vectors must be passed to 'fit'")
            predictions_unconverted = np.argmax(test_predicted_probabilities, axis=1)
            predictions = np.array([self.train_labels[x] for x in predictions_unconverted])

        # Fit the feature extractors
        self.feature_extractor.fit(x_train, y_train)

        # Collect the point wise features for test
        test_features, _ = self.feature_extractor.transform_test(x_test,
                                                                 predicted_probabilities=test_predicted_probabilities)

        y_meta = np.where(predictions == np.squeeze(y_test), 1, 0)

        for v in self.metamodels.values():
            # get the features from the dataframe necessary for metamodel
            mm_col_names = self._get_column_names(v["pointwise_features"], list(test_features.columns))
            mm_test_features = test_features[mm_col_names]

            # Now invoke the performance predictor
            v['perf_predictor'].fit(x_test, mm_test_features, y_meta)

        self.timer.stop('fit')

    def predict(self, x_prod, predicted_probabilities=None):
        self.timer.start('predict')
        result = {}

        if self.base_model is not None:
            predicted_probabilities = self.base_model.predict_proba(x_prod)
        else:
            try:
                assert predicted_probabilities is not None
            except:
                raise Exception(
                    "If base model is not provided to constructor, confidence vectors must be passed to 'predict'")

        prod_features, _ = self.feature_extractor.transform_prod(x_prod,
                                                                 predicted_probabilities,
                                                                 {})
        for v in self.metamodels.values():
            # get the features from the dataframe necessary for metamodel
            mm_col_names = self._get_column_names(v["pointwise_features"], list(prod_features.columns))
            mm_prod_features = prod_features[mm_col_names]

            # Get the PP predictions on prod
            v['predictor_output'] = v['perf_predictor'].predict(x_prod, mm_prod_features)

            v['accuracy'] = 100.0 * np.mean(v['predictor_output']['confidences'])

        confidences = None
        for v in self.metamodels.values():
            if confidences is None:
                confidences = v['predictor_output']['confidences']
            else:
                confidences = np.column_stack((confidences, v['predictor_output']['confidences']))
        result['pointwise_confidences'] = confidences if len(confidences.shape) == 1 else confidences.mean(axis=1)
        result['accuracy'] = np.mean([v['accuracy'] for v in self.metamodels.values()])

        self.timer.stop('predict')
        return result
