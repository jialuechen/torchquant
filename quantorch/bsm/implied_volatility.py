from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_LetsBeRational', [dirname(__file__)])
        except ImportError:
            import _LetsBeRational
            return _LetsBeRational
        if fp is not None:
            try:
                _mod = imp.load_module('_LetsBeRational', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _LetsBeRational = swig_import_helper()
    del swig_import_helper
else:
    import _LetsBeRational
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0



def norm_cdf(z):
    return _LetsBeRational.norm_cdf(z)
norm_cdf = _LetsBeRational.norm_cdf

def normalised_black_call(x, s):
    return _LetsBeRational.normalised_black_call(x, s)
normalised_black_call = _LetsBeRational.normalised_black_call

def normalised_vega(x, s):
    return _LetsBeRational.normalised_vega(x, s)
normalised_vega = _LetsBeRational.normalised_vega

def normalised_black(x, s, q):
    return _LetsBeRational.normalised_black(x, s, q)
normalised_black = _LetsBeRational.normalised_black

def black(F, K, sigma, T, q):
    return _LetsBeRational.black(F, K, sigma, T, q)
black = _LetsBeRational.black

def implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q):
    return _LetsBeRational.implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q)
implied_volatility_from_a_transformed_rational_guess = _LetsBeRational.implied_volatility_from_a_transformed_rational_guess

def implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price, F, K, T, q, N):
    return _LetsBeRational.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(price, F, K, T, q, N)
implied_volatility_from_a_transformed_rational_guess_with_limited_iterations = _LetsBeRational.implied_volatility_from_a_transformed_rational_guess_with_limited_iterations

def normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta, x, q, N):
    return _LetsBeRational.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta, x, q, N)
normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations = _LetsBeRational.normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations

def normalised_implied_volatility_from_a_transformed_rational_guess(beta, x, q):
    return _LetsBeRational.normalised_implied_volatility_from_a_transformed_rational_guess(beta, x, q)
normalised_implied_volatility_from_a_transformed_rational_guess = _LetsBeRational.normalised_implied_volatility_from_a_transformed_rational_guess