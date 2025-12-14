import re
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from astropy import constants as const
import astropy.units as u
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_EMISSION_LINES_FILE = PACKAGE_DIR / "data" / "emission_lines/emission_lines.csv"

LINE_GROUPS = {
    # Common doublets/multiplets
    '[OII]': ['[OII]-3727', '[OII]-3730'],
    '[NII]': ['[NII]-5756', '[NII]-6550', '[NII]-6585'],
    '[SII]': ['[SII]-6718', '[SII]-6733'],
    '[OIII]': ['[OIII]-4364', '[OIII]-4960', '[OIII]-5008'],
    '[NeIII]': ['[NeIII]-3870'],
    '[OI]': ['[OI]-5579', '[OI]-6302', '[OI]-6366'],
    '[SIII]': ['[SIII]-9071', '[SIII]-9533'],
    # Helium Lines
    'HeI': [
        'HeI-10833',
        'HeI-7067',
        'HeI-6680',
        'HeI-5877',
        'HeI-5017',
        'HeI-4923',
        'HeI-4473',
        'HeI-4122',
        'HeI-3890',
        'HeI-3873',
    ],
    'HeI_T': ['HeI-10833', 'HeI-7067', 'HeI-5877', 'HeI-3890'],
    'HeI_S': ['HeI-6680', 'HeI-5017', 'HeI-4923'],
    # Broad Lines (typically used for BLR)
    'HI': ['Ha', 'Hb', 'Hg', 'Hd', 'H7', 'H8', 'H9'],
    'HI': ['Ha', 'Hb', 'Hg', 'Hd'],
    'HeI': ['HeI-10833', 'HeI-7067', 'HeI-5877'],
    'OI': ['OI-1304', 'OI-8449'],
    'Pa': ['PaA', 'PaB', 'PaG', 'PaE', 'Pa9'],
    # Absorption Lines (typically stellar/ISM)
}


class EmissionLines:
    def __init__(self, filename: str | Path | None = None):
        if filename is None:
            filename = DEFAULT_EMISSION_LINES_FILE

        self.filename = Path(filename)
        self.table = Table.read(self.filename)
        self.lines = {l['key']: dict(l) for l in self.table}

        # Define common line groups
        self.groups = LINE_GROUPS

    # use get_line_nist to add new row
    # def get_line_nist(ion='H I', wave=[4000,6600], tolerance=1.0, single=False,
    #                  sortkey='Aki', clear_cache=True, verbose=False):
    def add_line(self, ion, **kwargs):
        self.table = vstack([self.table, get_line_nist(ion='H I', **kwargs)])
        self.table.sort('wave_vac')

    def search_line_nist(self, ion, **kwargs):
        return get_line_nist(ion=ion, **kwargs)

    def remove_key(self, key):
        ix_remove = np.where(self.table['key'] != key)[0]
        if len(ix_remove) > 0:
            self.table.remove_rows(ix_remove)

    def get_table(self, search_key=None, wave=None, multiplet=False, **kwargs):
        if wave is not None:
            iw = (self.table['wave_vac'] >= wave[0]) & (self.table['wave_vac'] <= wave[1])
        else:
            iw = self.table['wave_vac'] > 0

        if search_key is None:
            tab = self.table[iw]
        elif search_key in self.groups:
            # Create a mask for all keys in the alias list
            mask = np.zeros(len(self.table), dtype=bool)
            for key in self.groups[search_key]:
                mask |= self.table['key'] == key
            tab = self.table[mask & iw]
        elif search_key in self.table['key']:
            # Exact key match
            tab = self.table[(self.table['key'] == search_key) & iw]
        elif ' ' in search_key or ('[' in search_key and ']' in search_key):  # ions always have spaces
            # Try to format as "Element Roman" (e.g. "FeII" -> "Fe II")
            formatted_key = re.sub(r'(?<!\s)([IVX])', r' \1', search_key.replace(' ', ''), count=1)
            # Check if it matches 'ion' column directly or the formatted version
            ik = (self.table['ion'] == search_key) | (self.table['ion'] == formatted_key)
            tab = self.table[ik & iw]
        else:
            tab = self.table[(self.table['key'] == search_key) & iw]

        if multiplet:
            # Only recalculate if multiplet/line_ratio columns are missing or trivial
            has_multiplet = 'multiplet' in tab.colnames and np.any(tab['multiplet'] != 0)
            has_ratios = 'line_ratio' in tab.colnames and np.any(tab['line_ratio'] != 1.0)

            if not (has_multiplet and has_ratios):
                mr_kwargs = {k: v for k, v in kwargs.items() if k in ['Te', 'Ne', 'tolerance', 'verbose']}
                tab = multiplet_ratios(assign_multiplets(tab), **mr_kwargs)

        return tab

    def get_multiplet(self, key):
        ix = self.table['key'] == key
        if np.sum(ix) == 0:
            return None

        ion = self.table['ion'][ix][0]

        # Get all lines for this ion to correctly assign multiplets
        ion_tab = self.table[self.table['ion'] == ion]

        # Assign multiplets (this returns a copy/subset with 'multiplet' column populated)
        ion_tab = assign_multiplets(ion_tab)

        # Find the multiplet ID for the requested key
        match = ion_tab['key'] == key
        if np.sum(match) == 0:
            return None

        m_id = ion_tab['multiplet'][match][0]

        if m_id > 0:
            return ion_tab[ion_tab['multiplet'] == m_id]
        else:
            return ion_tab[match]

    def replace_ion(self, key, new_wave_mix, new_wave_vac, new_flux_ratio):
        #        self.remove_ion(ion)
        #        self.add_ion(ion, new_wave_mix, new_wave_vac, new_flux_ratio)
        pass

    def find_duplicates(self, keys=['ion', 'wave_vac'], remove=False):
        """
        Find duplicate entries in the table based on specified keys.

        Parameters:
            keys (list): List of column names to check for duplicates.
                         Default is ['ion', 'wave_vac'].
            remove (bool): If True, remove duplicates from the table, keeping only the first occurrence.

        Returns:
            Table: A table containing the duplicate entries.
        """
        from astropy.table import unique

        # Get groups of duplicates
        t = self.table.group_by(keys)

        # Find indices of groups with size > 1
        indices = t.groups.indices
        duplicates = []

        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            if end - start > 1:
                # This group has duplicates
                duplicates.append(t[start:end])

        if duplicates:
            dup_table = vstack(duplicates)

            if remove:
                n_before = len(self.table)
                self.table = unique(self.table, keys=keys)
                n_after = len(self.table)

                # Update the lines dictionary
                self.lines = {l['key']: dict(l) for l in self.table}
                print(f"Removed {n_before - n_after} duplicate rows.")

            return dup_table
        else:
            return None

    def add_lines(self, new_table, sort=True):
        """
        Add a table of new lines to the existing table.

        Parameters:
            new_table (Table): Table containing new lines to add.
            sort (bool): If True, sort the table by 'wave_vac' after adding.
        """
        self.table = vstack([self.table, new_table])

        if sort:
            self.table.sort('wave_vac')

        # Update the lines dictionary
        self.lines = {l['key']: dict(l) for l in self.table}

    def remove_lines(self, lines_to_remove):
        """
        Remove rows from the table that match the rows in lines_to_remove.

        Parameters:
            lines_to_remove (Table): Table containing lines to remove.
        """
        # We need to identify rows to remove.
        # We'll match on 'key', 'ion', and 'wave_vac' to be safe
        indices_to_remove = []

        for row in lines_to_remove:
            # Find matching rows in self.table
            mask = (
                (self.table['key'] == row['key'])
                & (self.table['ion'] == row['ion'])
                & (np.isclose(self.table['wave_vac'], row['wave_vac'], atol=1e-5))
            )

            indices = np.where(mask)[0]
            indices_to_remove.extend(indices)

        if indices_to_remove:
            indices_to_remove = np.unique(indices_to_remove)
            self.table.remove_rows(indices_to_remove)

            # Update the lines dictionary
            self.lines = {l['key']: dict(l) for l in self.table}

            print(f"Removed {len(indices_to_remove)} rows.")
        else:
            print("No matching rows found to remove.")

    def get_line_wavelengths(self, multiplet=True):
        lw = {row['key']: [row['wave_vac']] for row in self.table}
        lr = {row['key']: [row['line_ratio']] for row in self.table}
        if multiplet:
            multi = np.unique(self.table['multiplet'])
            for m in multi:
                ix = self.table['multiplet'] == m
                if self.table['ion'][ix][0] == 'H I':
                    k = 'Hydrogen'
                else:
                    k = (
                        self.table['key'][ix][0].split('-')[0]
                        + '-'
                        + ','.join([f'{w:.0f}' for w in self.table['wave_vac'][ix]])
                    )

                lw[k] = list(self.table['wave_vac'][ix])
                lr[k] = list(self.table['line_ratio'][ix])
                print(k, lw[k], lr[k])

        return lw, lr

    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        destination = Path(filename)
        if destination == DEFAULT_EMISSION_LINES_FILE:
            print(
                'Cannot overwrite default file emission_lines.csv, please specify new filename save(filename)'
            )

        self.table.write(destination, overwrite=True)

    def add_nist_columns(self, columns=['fik'], tolerance=1.0, verbose=False):
        """
        Add columns from NIST database to the emission lines table.

        Parameters:
            columns (list): List of column names to add from NIST (e.g., ['fik', 'Aki'])
            tolerance (float): Wavelength tolerance in Angstroms for matching lines
            verbose (bool): Print progress information

        Returns:
            Table: Updated table with new columns
        """
        from astroquery.nist import Nist
        import astropy.units as u

        # Make a copy of the table
        updated_table = self.table.copy()

        # Add columns if they don't exist
        for col in columns:
            if col not in updated_table.colnames:
                updated_table.add_column(Column(np.zeros(len(updated_table)), name=col))
                if verbose:
                    print(f"Added column '{col}' to table")

        # Process each row
        for i, row in enumerate(updated_table):
            ion = row['ion'].replace('[', '').replace(']', '')
            wave_vac = row['wave_vac']

            if verbose:
                print(f"Processing {i+1}/{len(updated_table)}: {ion} {wave_vac:.3f} Å")

            try:
                # Query NIST for the closest line
                nist_result = get_line_nist(
                    ion=ion,
                    wave=wave_vac,
                    tolerance=tolerance,
                    single=True,
                    clear_cache=False,
                    verbose=False,
                )

                if nist_result is not None and len(nist_result) > 0:
                    # Update requested columns
                    for col in columns:
                        if col in nist_result.colnames:
                            updated_table[col][i] = nist_result[col][0]
                            if verbose:
                                print(f"  Found {col} = {nist_result[col][0]}")
                        else:
                            if verbose:
                                print(f"  Warning: Column '{col}' not found in NIST result")
                else:
                    if verbose:
                        print(f"  No NIST match found within {tolerance} Å")

            except Exception as e:
                if verbose:
                    print(f"  Error querying NIST: {e}")
                continue

        return updated_table

    def regenerate_table(self, tolerance=1.0, verbose=False, skip_on_error=True):
        """
        Regenerate the emission lines table by querying NIST for each entry.

        This method creates a new table by searching NIST for the nearest line
        to each entry in the current table based on ion and wavelength.

        Parameters:
            tolerance (float): Wavelength tolerance in Angstroms for matching lines
            verbose (bool): Print progress information
            skip_on_error (bool): Skip entries that fail instead of raising an error

        Returns:
            Table: New table with updated entries from NIST
        """
        from astropy.table import vstack

        new_tables = []
        skipped = []

        for i, row in enumerate(self.table):
            ion = row['ion'].replace('[', '').replace(']', '')
            wave_vac = row['wave_vac']

            if verbose:
                print(f"Processing {i+1}/{len(self.table)}: {ion} {wave_vac:.3f} Å")

            try:
                # Query NIST for the closest line
                nist_result = get_line_nist(
                    ion=ion,
                    wave=wave_vac,
                    sortkey='Aki',
                    tolerance=tolerance,
                    single=True,
                    clear_cache=False,
                    verbose=False,
                )

                if nist_result is not None and len(nist_result) > 0:
                    new_tables.append(nist_result)
                    if verbose:
                        print(f"  Found: {nist_result['key'][0]} at {nist_result['wave_vac'][0]:.3f} Å")
                else:
                    skipped.append((ion, wave_vac, "No NIST match found"))
                    if verbose:
                        print(f"  No NIST match found within {tolerance} Å")

            except Exception as e:
                skipped.append((ion, wave_vac, str(e)))
                if verbose:
                    print(f"  Error querying NIST: {e}")
                if not skip_on_error:
                    raise

        # Stack all results into a single table
        if new_tables:
            regenerated_table = vstack(new_tables)
            regenerated_table.sort('wave_vac')

            if verbose:
                print(f"\nRegenerated table with {len(regenerated_table)} entries")
                if skipped:
                    print(f"Skipped {len(skipped)} entries:")
                    for ion, wave, reason in skipped:
                        print(f"  {ion} {wave:.3f} Å: {reason}")

            return regenerated_table
        else:
            if verbose:
                print("No entries could be regenerated from NIST")
            return None

    def to_unite(self, groups, save=False):
        """
        Produces unite style jsons from the line catalogs of el.table

        Parameters:
            groups (list): List of dictionaries defining the groups.
                           e.g. [{'name': 'default'}, {'emission1': '[OII],[NII]'}, ...]
            save (bool): If True, dump to json file named "{name}.json"

        Returns:
            dict: The unite style dictionary
        """
        import json
        import re
        from astropy.table import vstack

        unite_dict = {'Name': 'all', 'Unit': 'Angstrom', 'Groups': {}}
        # Add region if present

        for k, v in groups[0].items():
            print(k, v)
            unite_dict[k.capitalize()] = v

        for g in groups:
            # Handle the 'name' in the first element if present
            # Identify group name and line string
            group_name = None
            line_string = None
            kwargs = {}

            for k, v in g.items():
                if k in ['TieRedshift', 'TieDispersion', 'multiplet', 'Te', 'Ne', 'additional']:
                    kwargs[k] = v
                elif k == 'wave':
                    kwargs['wave'] = v
                elif isinstance(v, str):
                    group_name = k
                    line_string = v
                else:
                    # Assume other keys are kwargs too
                    kwargs[k] = v

            if group_name is None:
                continue

            # Auto-increment group name if no digits provided (e.g. 'emission' -> 'emission1')
            if not re.search(r'\d+$', group_name):
                base_name = group_name
                i = 1
                while True:
                    candidate = f"{base_name}{i}"
                    if candidate not in unite_dict['Groups']:
                        group_name = candidate
                        break
                    i += 1

            # Determine LineType (remove trailing digits)
            line_type = re.sub(r'\d+$', '', group_name)

            # Prepare Unite Group Config
            unite_group = {
                'TieRedshift': kwargs.pop('TieRedshift', True),
                'TieDispersion': kwargs.pop('TieDispersion', True),
                'Species': [],
            }

            # Prepare get_table kwargs
            # Default multiplet to True unless specified False
            multiplet = kwargs.pop('multiplet', True)
            additional = kwargs.pop('additional', None)

            # Fetch tables for each alias
            tables = []
            aliases = [x.strip() for x in line_string.split(',')]
            for alias in aliases:
                try:
                    t = None
                    # If multiplet is requested, check if alias is a specific line key that belongs to a multiplet
                    if multiplet and alias in self.table['key']:
                        t = self.get_multiplet(alias)

                    if t is None:
                        # Pass multiplet and kwargs (Te, Ne, etc.)
                        t = self.get_table(alias, multiplet=multiplet, **kwargs)
                    elif multiplet:
                        # If we got a table from get_multiplet, ensure ratios are calculated if missing
                        has_ratios = 'line_ratio' in t.colnames and np.any(t['line_ratio'] != 1.0)
                        if not has_ratios:
                            # Filter kwargs for multiplet_ratios
                            mr_kwargs = {
                                k: v for k, v in kwargs.items() if k in ['Te', 'Ne', 'tolerance', 'verbose']
                            }
                            t = multiplet_ratios(assign_multiplets(t), **mr_kwargs)

                    if len(t) > 0:
                        tables.append(t)
                except Exception as e:
                    print(f"Warning: Could not get table for alias '{alias}': {e}")

            if not tables:
                continue

            full_table = vstack(tables)

            # Group by Ion/Species
            ions = np.unique(full_table['ion'])

            for ion in ions:
                mask = full_table['ion'] == ion
                ion_table = full_table[mask]

                # Determine subgroups based on multiplets
                subgroups = []
                if multiplet and 'multiplet' in ion_table.colnames:
                    unique_multis = np.unique(ion_table['multiplet'])
                    # If 0 is present, those lines are treated individually or as a group 0
                    # But typically we want to iterate over all unique multiplet IDs found
                    for m in unique_multis:
                        subgroups.append(ion_table[ion_table['multiplet'] == m])
                else:
                    subgroups.append(ion_table)

                for species_table in subgroups:
                    # Normalize ratios per multiplet (weakest to 1.0)
                    if (
                        multiplet is not None
                        and 'multiplet' in species_table.colnames
                        and 'line_ratio' in species_table.colnames
                    ):
                        # Since we split by multiplet, we can just normalize the whole table if m > 0
                        m_vals = species_table['multiplet']
                        if len(m_vals) > 0 and m_vals[0] > 0:
                            ratios = species_table['line_ratio']
                            # Handle masked/nan
                            if hasattr(ratios, 'mask'):
                                valid_mask = ~ratios.mask & ~np.isnan(ratios)
                            else:
                                valid_mask = ~np.isnan(ratios)
                            # this may not be necessary but just in case
                            if np.any(valid_mask):
                                max_r = np.max(ratios[valid_mask])
                                if max_r > 0:
                                    species_table['line_ratio'] = ratios / max_r

                    # Format Species Name: 'H I' -> 'HI', '[O III]' -> '[OIII]'
                    species_name = ion.replace(' ', '')

                    # Append multiplet ID to species name if applicable
                    if multiplet and 'multiplet' in species_table.colnames:
                        m_vals = species_table['multiplet']
                        if len(m_vals) > 0 and m_vals[0] > 0:
                            species_name = f"{species_name}m{m_vals[0]}"

                    lines = []
                    for row in species_table:
                        wave = float(row['wave_vac'])

                        # Handle RelStrength
                        rel_strength = None
                        # Only use ratio if multiplet calculation was enabled
                        if multiplet and 'line_ratio' in row.colnames and row['multiplet'] > 0:
                            val = row['line_ratio']
                            if not np.ma.is_masked(val) and not np.isnan(val):
                                rel_strength = float(val)

                        lines.append({'Wavelength': wave, 'RelStrength': rel_strength})

                    species_entry = {'Name': species_name, 'LineType': line_type}
                    if additional:
                        if isinstance(additional, list):
                            species_entry['AdditionalComponents'] = {k: group_name for k in additional}
                        else:
                            species_entry['AdditionalComponents'] = {str(additional): group_name}
                    species_entry['Lines'] = lines
                    unite_group['Species'].append(species_entry)

            unite_dict['Groups'][group_name] = unite_group

        if save:
            filename = f"{unite_dict['Name']}.json"
            with open(filename, 'w') as f:
                json.dump(unite_dict, f, indent=4)

        return unite_dict


# Backwards compatible alias with the legacy implementation name.
# Emission_Lines = EmissionLines


# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion#
def vacuum_to_air(lambda_vac_in):
    lambda_vac = np.atleast_1d(lambda_vac_in)  # Ensure input is a numpy array for vectorized operations
    s2 = (1e4 / lambda_vac) ** 2

    # Compute the refractive index of air using the formula from Donald Morton (2000)
    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    lambda_air = lambda_vac / n_air
    lambda_air[lambda_vac < 2000] = lambda_vac[lambda_vac < 2000]

    if np.isscalar(lambda_vac_in):
        lambda_air = lambda_air.item()
    return lambda_air.tolist() if isinstance(lambda_vac_in, list) else lambda_air


# by N. Piskunov
def air_to_vacuum(lambda_air_in):
    # Ensure input is a numpy array for vectorized operations
    lambda_air = np.atleast_1d(lambda_air_in)
    s2 = (1e4 / lambda_air) ** 2

    # Compute the refractive index of air using the provided formula
    n_air = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s2)
        + 0.0001599740894897 / (38.92568793293 - s2)
    )
    lambda_vacuum = lambda_air * n_air
    lambda_vacuum[lambda_vacuum < 2000] = lambda_vacuum[lambda_vacuum < 2000]

    if np.isscalar(lambda_air_in):
        lambda_vacuum = lambda_vacuum.item()
    return lambda_vacuum.tolist() if isinstance(lambda_air_in, list) else lambda_vacuum

    # if np.iterable(lambda_vacuum):
    #     lambda_vacuum[lambda_vacuum < 2000] = lambda_air_arr[lambda_vacuum < 2000]
    #     return list(lambda_vacuum) if type(lambda_air) is list else lambda_vacuum
    # else:
    #     return lambda_air if lambda_vacuum < 2000 else lambda_vacuum


# test vacuum to air by converting to air and back, minus the lambda_vac from lambda_vac= 2000 to 1e5
# reproduces: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion?action=AttachFile&do=view&target=air2vac.gif
# reversible to within 1e-8
def test_vacuum_to_air():
    import matplotlib.pyplot as plt

    lambda_vac = np.linspace(1000, 1e5, 10000)
    lambda_air = vacuum_to_air(lambda_vac)
    lambda_vac_back = air_to_vacuum(lambda_air)
    plt.semilogx(lambda_vac, lambda_vac_back - lambda_vac)
    plt.ylim(-2e-8, 2e-8)


def roman_to_int(roman):
    roman_map = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
        'VIII': 8,
        'IX': 9,
        'X': 10,
        'XI': 11,
        'XII': 12,
        'XIII': 13,
        'XIV': 14,
        'XV': 15,
    }
    return roman_map.get(roman, None)  # Return None if not a valid numeral


def replace_with_latex(text):
    replacements = {
        'A': r'$\alpha$',
        'a': r'$\alpha$',
        'B': r'$\beta$',
        'b': r'$\beta$',
        'G': r'$\gamma$',
        'g': r'$\gamma$',
        'D': r'$\delta$',
        'd': r'$\delta$',
        'E': r'$\epsilon$',
        'e': r'$\epsilon$',
    }
    return ''.join(replacements.get(c, c) for c in text)


def construct_hydrogen_key(config):
    """
    Construct the 'key' for hydrogen entries based on configuration rules.
    """
    parts = config.split('-')
    if len(parts) != 2:
        return None  # Invalid configuration
    try:
        start = int(parts[0][0])  # first digit of first level: for Ly it is 1s, for H it is 2, etc.
        end = int(parts[1])
    except ValueError:
        return None  # Non-integer configuration

    diff = end - start
    series_map = {1: 'Ly', 2: 'H', 3: 'Pa', 4: 'Br', 5: 'Pf'}

    # Determine series
    if start in series_map:
        series = series_map[start]
        if start == 2:  # Special case for H series (2-level)
            special_names = ['a', 'b', 'g', 'd']
            if diff <= 4:
                return f"{series}{special_names[diff - 1]}"
            else:
                return f"{series}{end}"
        else:  # Use A, B, G, D, E for the first 5 entries, then numbers
            special_names = ['A', 'B', 'G', 'D', 'E']
            if diff <= 5:
                return f"{series}{special_names[diff - 1]}"
            else:  # Use numbers for others
                return f"{series}{end}"
    return None


def classify_transition(config, terms, Ji_Jk, nist_type=None, Aki=None, fik=None):
    """
    Classify a transition as forbidden, semi-forbidden, or permitted.
    Uses NIST type classification when available, with numerical overrides.
    """
    # Handle empty terms or Ji_Jk as permitted transitions (e.g., for H I)
    if not terms or not Ji_Jk:
        return "permitted"

    # Split inputs
    lower_config, upper_config = config.split('-')
    lower_term, upper_term = terms.split('-')
    lower_J, upper_J = map(lambda x: eval(x.strip()), Ji_Jk.split('-'))

    def get_parity(config):
        """
        Calculate parity based on the orbital contributions in the configuration.
        """
        l_values = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
        total_l = 0

        # Extract orbitals and their counts using regex
        orbitals = re.findall(r'([0-9]*)([spdfghi])([0-9]*)', config)
        for count_str, orbital, exponent_str in orbitals:
            count = int(count_str) if count_str else 1
            exponent = int(exponent_str) if exponent_str else 1
            if orbital in l_values:
                l = l_values[orbital]
                total_l += l * exponent
            else:
                print(f"Warning: Unknown orbital '{orbital}' in config '{config}', skipping")

        parity = 'even' if total_l % 2 == 0 else 'odd'
        return parity

    # Get parities
    lower_parity = get_parity(lower_config)
    upper_parity = get_parity(upper_config)

    def parse_term(term):
        """
        Parse a term string to extract spin multiplicity and orbital angular momentum.
        Handles NIST notation with prefixes (a, b, c, x, y, z, etc.) and asterisks.
        """
        # Remove leading lowercase letters (a, b, c, x, y, z, w, etc.) and asterisks
        # These are NIST labels for distinguishing terms, not part of the term symbol
        clean_term = re.sub(r'^[a-z]+', '', term.strip())
        clean_term = clean_term.rstrip('*')

        # Match: multiplicity + L symbol
        match = re.match(r'(\d+)([SPDFGHIKLMN]|\[\d+/?\d*\])', clean_term)
        if match:
            multiplicity = int(match.group(1))
            L_symbol = match.group(2)
            L_values = {
                'S': 0,
                'P': 1,
                'D': 2,
                'F': 3,
                'G': 4,
                'H': 5,
                'I': 6,
                'K': 7,
                'L': 8,
                'M': 9,
                'N': 10,
            }
            if L_symbol in L_values:
                L = L_values[L_symbol]
            elif L_symbol.startswith('[') and L_symbol.endswith(']'):
                L = float(L_symbol.strip('[]'))
            else:
                raise ValueError(f"Unknown term symbol: {L_symbol}")
            S = (multiplicity - 1) / 2
            return S, L
        else:
            raise ValueError(f"Invalid term format: {term}")

    # Extract term information
    try:
        S_lower, L_lower = parse_term(lower_term)
    except Exception as e:
        print(f"Warning: Failed to parse lower term '{lower_term}': {e}. Using default S=0, L=0.")
        S_lower, L_lower = 0, 0
    try:
        S_upper, L_upper = parse_term(upper_term)
    except Exception as e:
        print(f"Warning: Failed to parse upper term '{upper_term}': {e}. Using default S=0, L=0.")
        S_upper, L_upper = 0, 0

    # Calculate selection rules
    delta_S = abs(S_upper - S_lower)
    delta_L = abs(L_upper - L_lower)
    delta_J = abs(upper_J - lower_J)

    # Normalize NIST type
    nist_type = (nist_type or "").strip()
    if nist_type == "--":
        nist_type = "E1"

    # 0) If NIST explicitly labels a forbidden multipole, respect it
    if nist_type in {"M1", "E2", "M2", "E3"}:
        return "forbidden"

    # 1) If parity changes and NIST implies E1, treat as permitted/intercombination
    if lower_parity != upper_parity and nist_type in {"E1", ""}:
        return "semi-forbidden" if delta_S != 0 else "permitted"

    # 2) Numeric overrides: big A or f + parity change => effectively E1
    if lower_parity != upper_parity:
        if (Aki is not None and Aki >= 1e2) or (fik is not None and fik >= 1e-5):
            return "permitted"

    # 3) Classical fallbacks
    if lower_parity == upper_parity:
        return "forbidden"
    if delta_S != 0:
        return "semi-forbidden"
    if delta_J in {0, 1} and not (lower_J == 0 and upper_J == 0):
        return "permitted"
    return "forbidden"


def get_line_nist(
    ion='H I',
    wave=[4000, 6600],
    tolerance=1.0,
    single=False,
    sortkey=None,
    threshold=None,
    classification=None,
    clear_cache=False,
    multiplet=True,
    verbose=False,
    multiplet_lower_only=False,
    emissitivies=True,
    Te=10_000,
):
    from astroquery.nist import Nist
    import numpy.ma as ma

    if clear_cache:
        Nist.clear_cache()
    if np.iterable(wave):
        minwave, maxwave = wave
    else:
        minwave, maxwave = wave - tolerance, wave + tolerance

    ion = ion.replace(']', '').replace('[', '')
    print(f"Querying NIST for {ion} lines between {minwave} and {maxwave} Å")

    try:
        results = Nist.query(
            minwave << u.angstrom, maxwave << u.angstrom, linename=ion, wavelength_type='vacuum'
        )
    except Exception as e:
        print(f"Failed to query NIST: {e}")
        return None

    if results is None:
        print(f"No results found for {ion} in wavelength range {minwave}-{maxwave} Å")
        return None

    # Clean column names
    clean = lambda text: re.sub(r'\s+', '', text)
    results.rename_columns(results.colnames, [clean(c).replace('.', '') for c in results.colnames])

    # Filter hydrogen series
    if ion == 'H I':
        if verbose:
            print(results)
        ix = [str(r['Upperlevel']).split('|')[0].replace(' ', '').isdigit() for r in results]
        results = results[ix]

    # Helper function to check if value is masked/missing
    def is_masked_or_nan(val):
        """Check if a value is masked, None, empty string, or NaN."""
        if isinstance(val, ma.core.MaskedConstant):
            return True
        if val is None or val == '' or val == '--':
            return True
        try:
            return np.isnan(float(val))
        except (ValueError, TypeError):
            return False

    # Prepare output table
    output = Table(
        names=[
            'key',
            'ion',
            'wave_vac',
            'Ei',
            'Ek',
            'Aki',
            'fik',
            'gigk',
            'configuration',
            'terms',
            'Ji-Jk',
            'type',
            'references',
            'note',
            'wave_air',
            'ion_tex',
            'ion_tex_lambda',
            'classification',
        ],
        dtype=[
            'U14',
            'U9',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'U8',
            'U33',
            'U14',
            'U8',
            'U2',
            'U14',
            'U10',
            'float64',
            'U13',
            'U27',
            'U27',
        ],
    )

    for row in results:
        for col in results.colnames:
            # only operate on string-like columns
            if results[col].dtype.kind in ('U', 'S', 'O'):
                if not ma.is_masked(row[col]):
                    row[col] = re.sub(r'[\s?]+', '', row[col])

        row = dict(row)
        if verbose:
            print(row)
        ion_tab = ion

        def parse_level(level):
            """Parse level string, handling multiplet J values (e.g., '0,1,2')."""
            parts = [part.strip() for part in level.split('|')]
            if len(parts) == 3:
                config = parts[0]
                term = parts[1]
                J_str = parts[2]

                # Handle multiplet J values (e.g., '0,1,2')
                if ',' in J_str:
                    # Take the first J value for multiplets
                    J_str = J_str.split(',')[0].strip()

                return config, term, J_str
            else:
                raise ValueError(f"Invalid level format: {level}")

        # Parse Ei and Ek
        if is_masked_or_nan(row['EiEk']):
            continue
        energy_levels = row['EiEk'].split('-')
        Ei = float(energy_levels[0].strip().strip('[]').strip('()'))
        Ek = float(energy_levels[1].strip().strip('[]').strip('()'))

        # Handle NIST type
        nist_type = 'E1' if str(row['Type']) == '--' else str(row['Type'])

        # Construct key
        if ion == 'H I':
            lower = str(row['Lowerlevel']).split('|')[0].replace(' ', '')
            upper = str(row['Upperlevel']).split('|')[0].replace(' ', '')
            config = f"{lower}-{upper}"
            terms = JiJk = ''
            cval = 'permitted'
            key = construct_hydrogen_key(config)
            if 'gigk' not in row.keys():
                row['gigk'] = None
            if key is None:
                continue
        else:
            # Parse lower and upper levels
            lower_config, lower_terms, lower_J_str = parse_level(row['Lowerlevel'])
            upper_config, upper_terms, upper_J_str = parse_level(row['Upperlevel'])

            config = f"{lower_config}-{upper_config}"
            terms = f"{lower_terms}-{upper_terms}"
            JiJk = f"{lower_J_str}-{upper_J_str}"

            if verbose:
                print(config, terms, JiJk)

            # Extract Aki and fik values, handling masked data
            Aki_val = None if is_masked_or_nan(row['Aki']) else float(row['Aki'])
            fik_val = None if 'fik' not in row.keys() or is_masked_or_nan(row['fik']) else float(row['fik'])

            # Classify transition with NIST type and numerical data
            cval = classify_transition(config, terms, JiJk, nist_type=nist_type, Aki=Aki_val, fik=fik_val)

            # Ion label with brackets
            if cval == 'forbidden':
                ion_tab = f'[{ion}]'
            elif cval == 'semi-forbidden':
                ion_tab = f'{ion}]'

            # get rid of 8224.6+ notation in Ritz
            if type(row['Ritz']) is np.str_:
                row['Ritz'] = float(re.sub(r'[^\d.]', '', row['Ritz']))

            key = ion_tab.replace(' ', '') + f"-{row['Ritz']:.0f}"

        ion_tex = ion_tab.replace(']', '$]$').replace('[', '$[$')

        if ion == 'H I':
            ion_tex = ion_tex[:-1] + replace_with_latex(ion_tex[-1])

        if verbose:
            print(f"Classification: {cval}")

        # Convert vacuum to air wavelength
        wave_air = vacuum_to_air(row['Ritz'])

        # Output type defaults to NIST type
        out_type = nist_type if nist_type else 'E1'

        # Add row to output table
        output.add_row(
            [
                key.replace(' ', ''),
                ion_tab,
                round(row['Ritz'], 3),
                Ei,
                Ek,
                row['Aki'] if not is_masked_or_nan(row['Aki']) else 0.0,
                row['fik'] if 'fik' in row.keys() and not is_masked_or_nan(row['fik']) else 0.0,
                row['gigk'] if row['gigk'] is not None and not is_masked_or_nan(row['gigk']) else '',
                config.replace('.', ''),
                terms,
                JiJk,
                out_type,
                'NIST',
                '',
                wave_air,
                ion_tex,
                rf"{ion_tex}$\,\lambda{row['Ritz']:.0f}$".replace('$$', ''),
                cval,
            ]
        )

    # parse gigk like "8-10" into gi,gk and add gf = gi * fik
    if len(output) > 0:
        gf = np.full(len(output), np.nan)
        for i, (gi, fik) in enumerate(output['gigk', 'fik']):
            if ma.is_masked(gi) or ma.is_masked(fik):
                continue
            gf[i] = float(gi.split('-')[0]) * fik

        output['gf'] = gf

    # Filter by classification if requested
    if classification is not None and len(output) > 0 and ion != 'H I':
        output = output[output['classification'] == classification]

    if len(output) == 0:
        print(f"No matching lines found for {ion} in the specified range.")
        return None

    # support threshold strings like 'Ei<2.0' or 'Aki>1e3'
    if isinstance(threshold, str):
        if '>' in threshold:
            col, val = threshold.split('>')
            ix = output[col] > float(val)
        elif '<' in threshold:
            col, val = threshold.split('<')
            ix = output[col] < float(val)
        output = output[ix]

    if len(output) == 0:
        print(f"No matching lines found for {ion} in the specified range.")
        return None

    if sortkey is not None:
        if sortkey == 'wave_vac':
            dwave = np.abs(output['wave_vac'] - (wave if np.isscalar(wave) else np.mean(wave)))
            output['dwave'] = dwave
            output.sort('dwave')
            output.remove_column('dwave')
        else:
            print('sorting by ', sortkey)
            output.sort(sortkey)
            print(output['key', sortkey])
            if sortkey in ['Aki', 'fik', 'gf']:
                output.reverse()

    # e.g., Select highest Aki value
    if single:
        output = output[0:1]

    if multiplet:
        output = assign_multiplets(output, lower_only=multiplet_lower_only, verbose=verbose)

    if emissitivies:
        output = calculate_multiplet_emissivities(output, Te=Te, default=1.0)

    return output


# calc emission line ratios for a given ion at a given temperature and density
def emissivity_ratios(atom, level, wave_vac, Te=1e4, Ne=1e2, relative=False, tolerance=0.1, verbose=False):
    """
    Compute emissivity for one or more lines, given rest-frame vacuum wavelengths.

    Parameters
    ----------
    atom : str
        Element symbol, e.g. 'H', 'O', 'Fe'.
    level : int
        Ionization stage in PyNeb convention (1=I, 2=II, ...).
    wave_vac : float or array
        Vacuum wavelengths [Å] of the lines you care about.
    Te : float
        Electron temperature [K].
    Ne : float
        Electron density [cm^-3].
    relative : bool
        If True, normalize to max(emissivity) = 1 (ignoring NaNs).
    tolerance : float
        Max allowed |w_req - w_PyNeb| [Å] to consider a match.

    Returns
    -------
    emissivity : ndarray
        Emissivities for each input line (np.nan if no match within tolerance).
    """
    import pyneb as pn

    # ensure array handling
    wave_vac = np.atleast_1d(wave_vac)
    wave_air = vacuum_to_air(wave_vac)

    # Instantiate PyNeb atom
    if atom == 'H' or atom == 'He':
        neb_atom = pn.RecAtom(atom, level)
        neb_atom.case = 'B'
    else:
        neb_atom = pn.Atom(atom, level)

    # Get available line wavelengths (air Å)
    if neb_atom.wave_Ang is None:
        em = neb_atom.getEmissivity(Te, Ne)
        line_waves = np.array(list(em.keys()), dtype='float')
    else:
        line_waves = np.array(neb_atom.wave_Ang, dtype=float)
        if line_waves.ndim == 2:
            line_waves = line_waves[line_waves > 0]  # 1D array of all transition wavelengths

    emissivity = np.full(wave_air.shape, 0.0, dtype=float)

    for i, w in enumerate(wave_air):
        # Nearest PyNeb line
        if verbose:
            print(atom, level, 'line_waves', line_waves, f'{Te=} {Ne=}')
        idx = np.argmin(np.abs(line_waves - w))
        dw = abs(line_waves[idx] - w)

        if dw <= tolerance:
            w_neb = line_waves[idx]
            try:
                emissivity[i] = neb_atom.getEmissivity(Te, Ne, wave=w_neb)
            except Exception as e:
                print(f"Failed emissivity for {atom} {level} at {w_neb:.2f} Å: {e}")
        else:
            print(
                f"No PyNeb line for {atom} {level} within {tolerance:.2f} Å of {w:.2f} Å (closest {line_waves[idx]:.2f} Å, dw={dw:.2f} Å )"
            )

    # Normalize
    if relative:
        finite = emissivity > 0.0
        if finite.any():
            emissivity[finite] /= emissivity[finite].max()

    if verbose:
        print(f"Theoretical ratio {neb_atom.atom}", wave_vac, emissivity)
    return emissivity


def hydrogen_ratios(intab, wave=[2000, 1e5], Te=1e4, Ne=1e2, tolerance=1.0):
    tab = intab.copy()
    ix = np.where((tab['ion'] == 'H I') & (tab['wave_vac'] > wave[0]) & (tab['wave_vac'] < wave[1]))
    if not 'multiplet' in tab.colnames:
        tab['multiplet'] = 0
        tab['line_ratio'] = 0.0

    er = emissivity_ratios('H', 1, np.asarray(tab['wave_vac'][ix]), Te=Te, Ne=Ne, tolerance=tolerance)
    tab['line_ratio'][ix] = er
    if all(tab['multiplet'][ix] == 0):
        tab['multiplet'][ix] = np.max(tab['multiplet']) + 1
    else:
        tab['multiplet'][ix] = np.min(tab['multiplet'][ix])

    return tab


def assign_multiplets(tab, verbose=False, lower_only=False):
    """
    Assign multiplet numbers to transitions with the same configuration and term multiplicity.
    Multiplet numbers are unique within each ion, not globally.

    Parameters:
        tab: Table with columns 'ion', 'configuration', 'terms'
        verbose: Print diagnostic information
        lower_only: If True, group by lower term only (ignoring upper term)

    Returns:
        Table with added 'multiplet' and 'multiplet_key' columns
    """
    if 'configuration' not in tab.colnames:
        tab['configuration'] = [''] * len(tab)

    # Initialize multiplet column
    if 'multiplet' not in tab.colnames:
        tab['multiplet'] = 0

    if 'terms' not in tab.colnames:
        print("Table must hav 'terms' columns")
        return tab

    if lower_only:
        # Group by lower term only (extract before the '-')
        multiplet_keys = [
            (
                c
                if isinstance(t, np.ma.core.MaskedConstant)
                else c.split('-')[0].strip() + '_' + t.split('-')[0].strip() if '-' in t else t
            )
            for c, t in zip(tab['configuration'], tab['terms'])
        ]
    else:
        # Group by full configuration and term (both lower and upper)
        multiplet_keys = [
            (c if isinstance(t, np.ma.core.MaskedConstant) else c + '_' + t.strip())
            for c, t in zip(tab['configuration'], tab['terms'])
        ]

    tab['multiplet_key'] = multiplet_keys

    if verbose:
        print('multiplet_key', ' -- ', tab['multiplet_key'])

    # Group by ion first, then assign multiplets within each ion
    ion_groups = tab.group_by('ion')

    for ion_group in ion_groups.groups:
        # Group by multiplet_key within this ion
        multiplet_groups = ion_group.group_by('multiplet_key')

        multiplet_index = 1
        for ig, group in enumerate(multiplet_groups.groups):
            if len(group) > 1:
                if verbose:
                    print(
                        f"Multiplet {multiplet_index}: ion={group['ion'][0]}, "
                        f"config={group['multiplet_key'][0]} n={len(group['wave_vac'])}"
                    )

                # Get indices in original table
                for row in group:
                    mask = (tab['ion'] == row['ion']) & (tab['wave_vac'] == row['wave_vac'])
                    tab['multiplet'][mask] = multiplet_index

                multiplet_index += 1
            else:
                if verbose:
                    print(f"Single line: {group['ion'][0]} {group['multiplet_key'][0]}")

    del tab['multiplet_key']
    return tab


def calculate_multiplet_ratio(
    tab, ion, multiplet_number, Te=1e4, Ne=1e2, tolerance=0.1, default=1.0, verbose=False
):
    """
    Calculate line intensity ratios for a specific multiplet.

    Parameters:
        tab: Table with 'multiplet', 'ion', 'wave_vac' columns
        ion: Ion name (e.g., 'O III', '[O III]')
        multiplet_number: Multiplet identifier number
        Te: Electron temperature [K]
        Ne: Electron density [cm^-3]
        tolerance: Wavelength matching tolerance [Å]

    Returns:
        Array of line ratios for the multiplet transitions
    """
    # Select the multiplet
    mask = (tab['multiplet'] == multiplet_number) & (tab['ion'] == ion)
    group = tab[mask]

    if len(group) == 0:
        print(f"No multiplet found for ion={ion}, multiplet={multiplet_number}")
        return None

    # Extract atom and ionization level
    atom, level = ion.replace('[', '').replace(']', '').split(' ')

    if verbose:
        print(f"Calculating ratios for {atom} {level} multiplet {multiplet_number}")
        print(f"Wavelengths: {group['wave_vac']}")

    # Calculate emissivity ratios
    try:
        ratios = emissivity_ratios(
            atom,
            roman_to_int(level),
            np.asarray(group['wave_vac']),
            Te=Te,
            Ne=Ne,
            tolerance=tolerance,
            verbose=verbose,
        )
        # Check if ratios are all NaN or zero (which might happen if PyNeb returns nothing useful)
        if np.all(np.isnan(ratios)) or np.all(ratios == 0):
            raise ValueError("PyNeb returned no valid emissivities")

    except Exception as e:
        if verbose:
            print(
                f"PyNeb calculation failed for {ion}: {e}. Falling back to calculate_multiplet_emissivities."
            )
        # Fallback to 1:1
        ratios = np.full(len(group), default)

    return ratios


def multiplet_ratios(tab, Te=1e4, Ne=1e2, tolerance=0.1, verbose=False):
    """
    Calculate line intensity ratios for all multiplets in the table.
    Assumes multiplet column already exists and is populated.

    Parameters:
        tab: Table with 'multiplet', 'ion', 'wave_vac' columns already assigned
        Te: Electron temperature [K]
        Ne: Electron density [cm^-3]
        tolerance: Wavelength matching tolerance [Å]

    Returns:
        Table with updated 'line_ratio' column
    """
    if 'multiplet' not in tab.colnames:
        raise ValueError("Table must have 'multiplet' column. Run assign_multiplets() first.")

    # Add line_ratio column if it doesn't exist
    if 'line_ratio' not in tab.colnames:
        tab['line_ratio'] = 0.0

    unique_multiplets = np.unique(tab['multiplet'])
    #    unique_multiplets = unique_multiplets[unique_multiplets > 0]

    # Calculate ratios for each multiplet
    for multiplet_num in unique_multiplets:
        if multiplet_num == 0:
            continue

        # Get the ion for this multiplet
        mask = tab['multiplet'] == multiplet_num
        ion = tab['ion'][mask][0]

        # Calculate ratios
        ratios = calculate_multiplet_ratio(
            tab, ion, multiplet_num, Te=Te, Ne=Ne, tolerance=tolerance, verbose=verbose
        )

        if ratios is not None:
            tab['line_ratio'][mask] = ratios

    return tab


# def multiplet_ratios(tab, Te=1e4, Ne=1e2, tolerance=1.0):
#     ion = np.unique(tab['ion'])
#     # add H I to terms to get doublets / multiplets
#     # tab['multiplet_key'] = [
#     #     c + '_' + t for c, t in zip(tab['configuration'], tab['terms'])
#     #     c + '_' + t for c, t in zip(tab['configuration'], tab['terms'])
#     # ]
#     tab['multiplet_key'] = [
#         c if isinstance(t, np.ma.core.MaskedConstant) else c + '_' + re.sub(r'(\d+)[A-Za-z0-9*]*', r'\1', t)
#         for c, t in zip(tab['configuration'], tab['terms'])
#     ]

#     print('multiplet_key', ' -- ', tab['multiplet_key'])

#     grouped_table = tab.group_by(['ion', 'multiplet_key'])
#     grouped_table['multiplet'] = 0
#     grouped_table['line_ratio'] = 0.0
#     multiplet_index = 1
#     for ig, group in enumerate(grouped_table.groups):
#         print(
#             ig,
#             f"Group: ion={group['ion'][0]}, config={group['multiplet_key'][0]} wlen={len(group['wave_vac'])}",
#         )
#         print('GROUP len', len(group))

#         if len(group) > 1:
#             atom, level = group['ion'][0].replace('[', '').replace(']', '').split(' ')
#             print('GROUP', atom, level, group['wave_vac'])
#             er = emissivity_ratios(
#                 atom, roman_to_int(level), np.asarray(group['wave_vac']), Te=Te, Ne=Ne, tolerance=tolerance
#             )

#             start, end = grouped_table.groups.indices[ig], grouped_table.groups.indices[ig + 1]
#             grouped_table['multiplet'][start:end] = multiplet_index
#             grouped_table['line_ratio'][start:end] = er
#             multiplet_index += 1
#         else:
#             print('SKIP', group['ion'][0], group['multiplet_key'][0])

#     #    grouped_table = hydrogen_ratios(grouped_table,  Te=Te, Ne=Ne, tolerance=tolerance)
#     #    grouped_table.remove_columns(['multiplet_key'])
#     grouped_table.sort('wave_vac')

#     return grouped_table

# adopted from pyqso
continuum_windows = [
    (1150.0, 1170.0),
    (1275.0, 1290.0),
    (1350.0, 1360.0),
    (1445.0, 1465.0),
    (1690.0, 1705.0),
    (1770.0, 1810.0),
    (1970.0, 2400.0),
    (2480.0, 2675.0),
    (2925.0, 3400.0),
    (3775.0, 3832.0),
    (4000.0, 4050.0),
    (4200.0, 4230.0),
    (4435.0, 4640.0),
    (5100.0, 5535.0),
    (6005.0, 6035.0),
    (6110.0, 6250.0),
    (6800.0, 7000.0),
    (7160.0, 7180.0),
    (7500.0, 7800.0),
    (8050.0, 8150.0),
]


def replace_greek(text, tex=True):
    # Dictionary mapping Unicode Greek letters to LaTeX representations
    greek_to_latex = {
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "γ": r"$\gamma$",
        "δ": r"$\delta$",
        "ε": r"$\epsilon$",
        "ζ": r"$\zeta$",
        "η": r"$\eta$",
        "θ": r"$\theta$",
        "ι": r"$\iota$",
        "κ": r"$\kappa$",
        "λ": r"$\lambda$",
        "μ": r"$\mu$",
        "ν": r"$\nu$",
        "ξ": r"$\xi$",
        "ο": r"$o$",
        "π": r"$\pi$",
        "ρ": r"$\rho$",
        "σ": r"$\sigma$",
        "τ": r"$\tau$",
        "υ": r"$\upsilon$",
        "φ": r"$\phi$",
        "χ": r"$\chi$",
        "ψ": r"$\psi$",
        "ω": r"$\omega$",
        "Α": r"$\Alpha$",
        "Β": r"$\Beta$",
        "Γ": r"$\Gamma$",
        "Δ": r"$\Delta$",
        "Ε": r"$E$",
        "Ζ": r"$Z$",
        "Η": r"$H$",
        "Θ": r"$\Theta$",
        "Ι": r"$I$",
        "Κ": r"$K$",
        "Λ": r"$\Lambda$",
        "Μ": r"$M$",
        "Ν": r"$N$",
        "Ξ": r"$\Xi$",
        "Ο": r"$O$",
        "Π": r"$\Pi$",
        "Ρ": r"$P$",
        "Σ": r"$\Sigma$",
        "Τ": r"$T$",
        "Υ": r"$\Upsilon$",
        "Φ": r"$\Phi$",
        "Χ": r"$X$",
        "Ψ": r"$Ψ$",
        "Ω": r"$\Omega$",
    }

    # Replace each Greek letter in the text
    if tex:
        for greek, latex in greek_to_latex.items():
            text = text.replace(greek, latex)
    else:
        for greek, latex in greek_to_latex.items():
            text = text.replace(greek, '')

    return text


def replace_brackets_with_dollars(text):
    text = text.replace("[", r"$[$")
    text = text.replace("]", r"$]$")
    return text.replace('$$', '')


from astropy.table import MaskedColumn, Table, vstack

# obsolete get from NIST
# def hydrogen_lines(
#         tab_input='/Users/ivo/Desktop/current/agn/agn/data/emission_lines/table_4_wiese_2009.csv',
#         ndigits=3):
#     htab = Table.read(tab_input)
#     htab['references'] = 'W09'

#     #transition -> configuration
#     #lambda_vac, Ei, Ek, loggf
#     for c in ['wave_air', 'wave_vac', 'Ei', 'Ek', 'loggf']:
#         htab[c] = np.char.replace(htab[c].data, ' ', '')

#     # replace cm^-1 with Angstrom
#     htab['wave_vac'] = [
#         round(1.0 / float(r.split('cm')[0].replace(' ', '')) *
#               u.cm.to(u.Angstrom), ndigits)  # from cm^-1 to Angstrom
#         if 'cm' in r else r for r in htab['wave_vac'].data
#     ]

#     # replace cm^-1 with eV
#     for c in ['Ei', 'Ek']:
#         htab[c] = [
#             round(float(r) * 1.239841e-4, ndigits) for r in htab[c].data
#         ]

#     for c in ['wave_air', 'wave_vac', 'Aki', 'loggf']:
#         htab[c] = htab[c].astype(float)

#     return htab

# const.h * const.c / const.e * 106632.158
# E(eV) = h*c/e * E(cm-1)
#      = 1.2398419843320026e-4 * E(cm-1)
# * 1.2398419843320026e-4
# (1/5331.596 << u.cm).to(u.angstrom)


def add_hydrogen_entries(emission_table, hydrogen_table, tolerance=0.27):
    """
    Replace existing hydrogen entries in the emission line table with entries from the hydrogen table.

    Parameters:
        emission_table (Table): The original emission line table.
        hydrogen_table (Table): The table containing hydrogen line data.
        tolerance (float): The wavelength tolerance (in Å) for matching entries.

    Returns:
        Table: The updated emission line table.
    """

    # Mask to mark rows for removal in the emission table
    remove_index = []
    updated_emission_table = emission_table.copy()

    # Process each row in the hydrogen table
    for h_row in hydrogen_table:
        h_wave = h_row['wave_vac']
        config = h_row['configuration']
        key = construct_hydrogen_key(config)

        # Check for a matching entry in the emission table
        match_found = False
        for i, e_row in enumerate(emission_table):
            e_wave = e_row['wave_vac']
            if abs(h_wave - e_wave) <= tolerance:
                remove_index.append(i)
                break

        updated_emission_table.add_row(h_row)

    # Remove the marked rows
    updated_emission_table.remove_rows(remove_index)
    updated_emission_table.sort('wave_vac')

    return updated_emission_table


# missing are optical Ca lines
# ------------------------------------------------------------ first time generate table
from astropy.table import Column


def generate_line_table(
    tab_input='/Users/ivo/Desktop/current/agn/agn/data/emission_lines/emission_lines.csv', ndigits=3
):
    tab = Table.read(tab_input)

    # add Aki column for transition probabilities
    tab.add_column(Column(np.zeros(len(tab)), name='Aki'), index=4)

    b_air = tab['wave_mix'] > 2000
    tab['wave_vac'] = tab['wave_mix']
    tab['wave_vac'][b_air] = np.round(air_to_vacuum(tab['wave_mix'])[b_air], ndigits)
    tab['wave_air'] = MaskedColumn(tab['wave_mix'], mask=~b_air)

    for t in [tab]:
        for c in t.colnames:
            if isinstance(t[c], MaskedColumn):
                if np.issubdtype(t[c].dtype, str):
                    t[c][t[c].mask] = ''
                if np.issubdtype(t[c].dtype, float):
                    t[c][t[c].mask] = 0.0

    tab['type'][tab['type'].mask] = 'E1'
    for c in ['configuration', 'terms', 'Ji-Jk']:
        tab[c] = [
            #            r if isinstance(r, np.ma.core.MaskedConstant) else r.encode(
            r.encode('ascii', 'ignore').decode('ascii').replace(' ', '')
            for r in tab[c]
        ]

    # add hydrogen lines
    htab = hydrogen_lines(ndigits=ndigits)

    # cross check with grizli table

    tab['ion_tex'] = [replace_greek(replace_brackets_with_dollars(i)) for i in tab['ion']]
    tab['ion_tex_lambda'] = [
        (i + r'$\,\lambda' + f'{w:.0f}$').replace('$$', '') for i, w in zip(tab['ion_tex'], tab['wave_vac'])
    ]

    tab['key'] = [
        replace_greek(i, tex=False).replace(' ', '') + f'-{w:.0f}'
        for i, w in zip(tab['ion'], tab['wave_vac'])
    ]

    tab.remove_columns(['wave_mix', 'creationip'])
    for c in ['wave_vac', 'ion', 'key']:
        tab.columns.move_to_end(c, last=False)

    # add hydrogen lines
    htab = get_line_nist(ion='H I', wave=[900, 1e5])
    tab = add_hydrogen_entries(tab, htab, tolerance=0.27)

    # add grizli lines
    tab = add_grizli_lines(tab)

    # add multiplet ratios
    mtab = multiplet_ratios(tab)

    mtab.write(DEFAULT_EMISSION_LINES_FILE, overwrite=True)
    return mtab


# agn.emission_lines.get_line_nist('O I', agn.emission_lines.air_to_vacuum(7990), single=True)
# mtab = agn.emission_lines.multiplet_ratios(tab)


def add_grizli_lines(tab):
    # no match HeII-5412 H F [5412.5] nearest [FeVI]-5426 5425.728 # not in NIST
    # no match MgII M M [2799.117] nearest MgII-2796 2796.352 -> [MgII]-2803 already in table
    # no match SiIV+OIV-1398 S O [1398.0] nearest OIV]-1397 1397.232 -> [OIV]-1398 already in table
    # no match NI-5199 N F [5199.4] nearest FeII-5199 5199.024 -> [NI]-5202 already in table
    # no match NIII-1750 N N [1750.0] nearest NIII]-1749 1748.656 -> [NIII]-1750 already in table
    # no match NV-1240 N N [1240.81] nearest NV-1239 1238.821 -> [NV]-1240 already in table
    lines = {
        'O I': [5578, 7990, 11290.00, 13168.4, 7777.5],
        'He I': [6680, 10832],
        'Ne IV': [2425, 2422],
        'Na I': [5891, 5897],
        'Ca II': [3934.78, 3969.591],
    }

    # Loop through the dictionary and call the function
    ion_list = [
        get_line_nist(ion, w, tolerance=1, single=True)
        for ion, wavelengths in lines.items()
        for w in wavelengths
    ]

    new_tab = vstack([tab] + ion_list)
    #    new_tab.sort('wave_vac')
    return new_tab


def compare_grizli_entries(tab, tolerance=1.0):
    from . import models

    lw, lr = models.get_line_wavelengths()
    # double check existence of each single length line entry in the grizli lw, lr dictionaries
    element = np.asarray([i.replace('[', '').replace(']', '')[0] for i in tab['key']])
    wave = np.asarray(tab['wave_vac'])

    for k in lw:
        if len(lw[k]) > 1:
            continue
        dw = lw[k][0] - wave
        imin = np.argmin(np.abs(dw))
        #        print(k, lw[k], element[imin], tab['ion'][imin], element[imin] == k[0], dw[imin], dw[imin] < tolerance)
        grizli_key = k.replace('[', '').replace(']', '')
        if (element[imin] == k[0]) and (dw[imin] < tolerance):
            print(f' match {k} {k[0]} {lw[k][0]} {tab['key'][imin]} {tab['wave_vac'][imin]}')
        else:
            print(
                'no match',
                k,
                k[0],
                element[imin],
                lw[k],
                f'nearest {tab['key'][imin]} {tab['wave_vac'][imin]}',
            )
    return element


# get nearest value in dict of lists.
# search among the first element of the list
# only consider lists with <= max_len
def find_nearest_key(lw_dict, value, min_len=1, max_len=3, **kwargs):
    singledict = single_line_dict(lw_dict, min_len=1, max_len=3, atol=0.1)
    #    print(singledict['OI-6302'])
    #    singledict = {k: v[0] for k, v in dictionary.items() if len(v) >= min_len and len(v) <= max_len}
    # singledict = {k: v[0] for k, v in dictionary.items() if len(v) == min_len}
    return min(singledict.keys(), key=lambda k: abs(singledict[k][0] - value))


# def get_line_keys(lw, line_complex, **kwargs):
#    return [models.find_nearest_key(lw, k) for k in lw[line_complex]]


def get_line_list():
    lw, lr = get_line_wavelengths()
    ln = {k: get_line_keys(lw, k) for k in lw}
    return lw, lr, ln


def unique_lines():
    lw, lr = get_line_wavelengths()
    uw = list(set([w for k in lw for w in lw[k]]))
    uw.sort()
    un = [find_nearest_key(lw, w) for w in uw]
    return {find_nearest_key(lw, w): w for w in uw}, uw, un


def cdf(wave, flux):
    norm = np.trapezoid(flux, wave)
    if norm == 0:
        return np.zeros_like(wave)
    else:
        return np.cumsum(flux / np.trapezoid(flux, wave) * np.gradient(wave))


def calculate_multiplet_emissivities(tab, Te=10_000, default=1.0, verbose=False):
    """
    Calculate relative line emissivities for multiplets using optically-thin approximation.

    For permitted lines sharing the same lower level:
        I ∝ (g_u * f_lu / λ³) * exp(-E_u / kT)

    Line ratios within a multiplet (same lower level):
        I₁/I₂ = (λ₂/λ₁)³ * (f₁/f₂) * (g_u1/g_u2) * exp[-(E_u1 - E_u2) / kT]

    Parameters
    ----------
    tab : Table
        Must contain: 'ion', 'multiplet', 'wave_vac', 'Ek' (upper energy in eV),
        and either 'gf' or ('fik' and 'gigk')
    Te : float
        Excitation temperature [K]
    verbose : bool
        Print diagnostic information

    Returns
    -------
    Table
        Input table with added 'line_ratio' column (normalized within each multiplet)
    """
    KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K

    tab = tab.copy()

    # Add line_ratio column if it doesn't exist
    if 'line_ratio' not in tab.colnames:
        tab['line_ratio'] = default

    # Parse gigk to get g_u (upper state degeneracy)
    def parse_gigk(gigk_str):
        """Parse gigk string like '10-8' to get (g_i, g_k)."""
        if not gigk_str or gigk_str == '':
            return np.nan, np.nan
        try:
            parts = gigk_str.split('-')
            gi = float(parts[0])  # lower state
            gk = float(parts[1])  # upper state
            return gi, gk
        except (ValueError, IndexError):
            return np.nan, np.nan

    # Extract gi, gu for all rows
    gi_all = []
    gu_all = []
    for g in tab['gigk']:
        gi, gu = parse_gigk(str(g))
        gi_all.append(gi)
        gu_all.append(gu)

    gi_all = np.array(gi_all)
    gu_all = np.array(gu_all)

    # Get oscillator strength f
    if 'gf' in tab.colnames and np.any(tab['gf'] > 0):
        f_all = tab['gf'].astype(float) / gi_all
    elif 'fik' in tab.colnames:
        f_all = tab['fik'].astype(float)
    else:
        if verbose:
            print("Warning: No 'gf' or 'fik' column found")
        f_all = np.ones(len(tab))

    # Group by ion and multiplet
    ion_groups = tab.group_by(['ion', 'multiplet'])

    for group in ion_groups.groups:
        if len(group) <= 1 or group['multiplet'][0] == 0:
            continue

        ion = group['ion'][0]
        multiplet_num = group['multiplet'][0]

        # Get indices in original table
        mask = (tab['ion'] == ion) & (tab['multiplet'] == multiplet_num)
        idx = np.where(mask)[0]

        # Extract parameters for this multiplet
        lam = np.array(group['wave_vac'], dtype=float)  # Å
        Eu = np.array(group['Ek'], dtype=float)  # eV (upper level energies)
        gu = gu_all[idx]
        f = f_all[idx]

        # Calculate relative intensities using energy differences from first line
        # Reference line (usually lowest energy or first in multiplet)
        E_ref = Eu[0]

        # I_i / I_ref = (λ_ref/λ_i)³ * (f_i/f_ref) * (g_ui/g_u_ref) * exp[-(E_ui - E_u_ref) / kT]
        delta_E = Eu - E_ref  # Energy difference from reference

        I_weight = (lam[0] / lam) ** 3 * (f / f[0]) * (gu / gu[0]) * np.exp(-delta_E / (KB_EV * Te))

        # Normalize within multiplet
        if np.sum(I_weight) > 0:
            I_norm = I_weight / np.sum(I_weight)
        else:
            I_norm = np.zeros_like(I_weight)

        # Assign back to table
        tab['line_ratio'][idx] = I_norm

        if verbose:
            print(f"\n{ion} multiplet {multiplet_num}:")
            print(f"  Wavelengths: {lam}")
            print(f"  Upper energies: {Eu} eV")
            print(f"  ΔE from ref: {delta_E} eV")
            print(f"  Relative intensities: {I_norm}")

    return tab
