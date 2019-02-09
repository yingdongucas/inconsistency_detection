two_version_range_keyword_list = [(' before ', '<=', '<='), (' prior to ', '<=', '<='), (' up to ', '<=', '<='),
                     (' through ', '<=', '<='), (' to ', '<=', '<='),
                     (' - ', '<=', '<='),  (' < ', '<=', '<=')
                     ]

one_version_range_keyword_list = [(' and possibly earlier', '<='),
                     (' and earlier', '<='), (' or earlier', '<='), (' earlier', '<='), (' and earler', '<='),
                     (' and prior', '<='), (' or prior', '<='), (' prior', '<='), ('prior to ', '<'), ('prior ', '<'),
                     (' and older', '>='), (' or older', '>='),
                     (' and higher', '>='), (' or higher', '>='),
                     (' and later', '>='), (' or later', '>='),
                     (' and greater', '>='), (' or greater', '>='),
                     ('before ', '<'), (' before', '<'), (' and before', '<='), (' or before', '<='),
                     ('prior ', '<='),
                     ('through ', '<='), (' through', '<='),
                     (' and lower', '<='), (' or lower', '<='),
                     (' and below', '<='), (' or below', '<='),
                     (' and upper', '>='),   (' or upper', '>='),
                     ('up to ', '<='), ('up to and including ', '<='),
                     ('<= ', '<='), ('>= ', '>='),
                     ('< = ', '<='), ('> = ', '>='),
                     ('< =', '<='), ('> =', '>='),
                     ('< ', '<='), ('> ', '>='),
                     (' onwards', '>=')
                     ]

opposite_symbol_dict = {'<': '>', '<=': '>=', '>':'<', '>=':'<='}
version_regex = r'(v)?[\d]{1,7}((\.[\d]{1,7}){1,7}(\.x)?|\.x)'