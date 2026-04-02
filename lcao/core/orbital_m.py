def normalize_orbital_m(raw_m, l, *, source='ORB_INDX', orbital_index=None, file_path=None):
    """Normalize magnetic quantum number encoding to signed ``m``.

    Accepted encodings:
    - signed encoding: ``m in [-l, ..., +l]`` (used as-is)
    - legacy encoding: ``ml in [1, ..., 2l+1]`` (converted by ``ml - l - 1``)
    """
    m_value = int(raw_m)
    l_value = int(l)

    if -l_value <= m_value <= l_value:
        return m_value

    if 1 <= m_value <= (2 * l_value + 1):
        return m_value - l_value - 1

    context = _orbital_context(source=source, orbital_index=orbital_index, file_path=file_path)
    raise ValueError(
        f'Invalid magnetic quantum number encoding: m={m_value}, l={l_value}{context}. '
        f'Expected signed [-l..l] or legacy [1..2l+1].'
    )


def validate_signed_orbital_m(m, l, *, source='ORB_INDX', orbital_index=None, file_path=None):
    m_value = int(m)
    l_value = int(l)
    if abs(m_value) <= l_value:
        return

    context = _orbital_context(source=source, orbital_index=orbital_index, file_path=file_path)
    raise ValueError(
        f'Invalid signed magnetic quantum number before Yml: m={m_value}, l={l_value}{context}. '
        'Expected abs(m) <= l.'
    )


def _orbital_context(*, source, orbital_index, file_path):
    details = [f' source={source}']
    if file_path is not None:
        details.append(f', file={file_path}')
    if orbital_index is not None:
        details.append(f', orbital_index={int(orbital_index)}')
    return ''.join(details)
