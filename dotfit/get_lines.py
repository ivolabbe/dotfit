# %%
from dataclasses import dataclass
from pathlib import Path
import importlib
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from astropy import conf
from astropy import units as u
from astropy.table import Table, vstack
from astroquery.nist import Nist
from astropy.utils.data import download_file

import dotfit

from dotfit.emission_lines import (
    EmissionLines,
    get_line_nist,
    plot_ion_models,
    plot_lines,
    assign_multiplets,
    compute_line_ratios,
)

conf.max_lines = 1000  # default number of rows astropy will print
conf.max_width = 200


DJA_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/'


def load_monster_spectra(filenames, redshift, base_url=DJA_URL):
    """Load multiple NIRSpec gratings for a single source."""
    root = filenames[0].split('_')[0]
    monster = {'z': redshift}

    for fname in filenames:
        grating = fname.split('_')[1].split('-')[0]
        url = f"{base_url}{root}/{fname}"
        local_path = download_file(url, cache=True)
        spec = Table.read(local_path, 'SPEC1D')

        fλ_unit = u.erg / u.s / u.cm**2 / u.AA
        wave = spec['wave'].to(u.AA)
        flux = spec['flux'].to(fλ_unit, equivalencies=u.spectral_density(wave))
        err = spec['err'].to(fλ_unit, equivalencies=u.spectral_density(wave))
        wave = wave / (1 + redshift)

        monster[grating] = {'url': url, 'grating': grating, 'wave': wave, 'flux': flux, 'err': err}

    return monster


filenames = [
    'abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits',
    'abell2744-greene-v4_g395m-f290lp_8204_45924.spec.fits',
    'abell2744-greene-v4_prism-clear_8204_45924.spec.fits',
]

monster = load_monster_spectra(filenames, redshift=4.465)
datadir = Path(dotfit.__file__).parent / 'data/emission_lines/'

el = EmissionLines()

# from astropy.units import Quantity
# Quantity(spec['wave'])
# tab = el.table[el.table['Ei'] < 6]
# plot_ion_models(tab=tab,wave_range=(3100,4200),spectrum_dict=monster,spectrum_key='g235m',legend=False)

# from dotfit import LineExplorer

# app = LineExplorer(
#     spectrum_dict=spectrum_dict,
#     emission_lines=tab,
#     redshift=4.465,  # Your object's redshift
#     object_name="abell2744-8204-45924",
# )

# In a Jupyter notebook:
# app.panel()

# tab = assign_multiplets(el.table, verbose=True)
# tab = compute_line_ratios(el.table, Te=10_000, Ne=1000, verbose=True)
# tab.write('emission_lines.csv', overwrite=True)

# tab_amr = apply_multiplet_rules(el.table, verbose=True)
# tab_amr
# tab_amr.write('emission_lines.csv', overwrite=True)


# tab_amr = apply_multiplet_rules(el.table, verbose=True)
# tab_amr[150:162]

# tab = apply_multiplet_rules(el.get_table('[Fe II]'), verbose=True)
# tab_amr = apply_multiplet_rules(el.table[0:162], verbose=True)
# tab.write('emission_lines.csv', overwrite=True)
# tab = el.table
# del tab['note']
# tab['note'] = '--'
# tab_amr.write('emission_lines.csv', overwrite=True)
# el.table = tab_amr

# caii = el.get_multiplet('CaII-3935')
# tab_neiii = el.get_multiplet('[NeIII]-3870')
# apply_multiplet_rules(tab_neiii, verbose=True)
# apply_multiplet_rules(el.table[10], verbose=True)
# apply_multiplet_rules(el.get_table('Ca II'), verbose=True)
# el.table[150:162] = apply_multiplet_rules(el.table[150:162], verbose=True)
# @@@ FIX THIS
# el.table.write('emission_lines.csv', overwrite=True)
# el.table = tab_amr
# el.get_table('[O III]')

# -------------------------------------------- get optical only
el = EmissionLines('feti_kurucz.csv')
opt = el.get_table(wave=[3700, 7100])
plt.hist(np.log10(opt['line_ratio']), bins=50, range=(-10, 1))

opt = el.get_table(wave=[3700, 7100], filter='line_ratio>1e-2')
plt.hist(np.log10(opt['line_ratio']), bins=50, range=(-10, 1))

opt.write(datadir / 'fe_optical.csv', overwrite=True)

# plt.hist(np.log10(opt['gf']), bins=50)

# %% ----------------------------------------------------------------------------------
# Fe II, Fe I lines absorption from Kurucz
from pathlib import Path
import dotfit
import numpy as np
from dotfit.emission_lines import (
    read_kurucz_table,
    EmissionLines,
    assign_multiplets,
    calculate_multiplet_emissivities,
)
from astropy import conf
from astropy.table import Table, vstack

conf.max_lines = 4000  # default number of rows astropy will print
conf.max_width = 200


def prep_table(fname, gf_threshold=1e-5, Te=5_000):
    kfile = Path(dotfit.__file__).parent / 'data/emission_lines/pool/' / fname
    tab = read_kurucz_table(str(kfile))
    tab = assign_multiplets(tab, lower_only=True)
    tab = calculate_multiplet_emissivities(tab, Te=Te)
    return tab


tabnames = ['kurucz_FeI.dat', 'kurucz_FeII.dat', 'kurucz_TiI.dat', 'kurucz_TiII.dat']
tab_fe = vstack([prep_table(fname, gf_threshold=1e-5) for fname in tabnames])
tab_fe.sort('wave_vac')

el = EmissionLines()
el.table = tab_fe
el.table.write('feti_kurucz.csv', overwrite=True)

# %%

# # tab = el.get_table('Fe II', wave=[4400, 4700])
# datadir = Path(dotfit.__file__).parent / 'data/emission_lines/'
# tab = Table.read(datadir / 'fe_kurucz.csv')
# idx = (tab['Ei'] < 4) & (tab['gf'] > 1e-5)  # & (tab['multiplet_term'] == 'b4F')
# plot_lines(tab[idx], fwhm_kms=300.0, legend=True, Te=10_000)


## TODO add Ca II, Mg I, K I lines, C I, CII lines
# https://github.com/yi-jia-li/cue/blob/main/src/cue/data/lineList.dat
# check multiplets of these transition:
# Ne [NeIV]-2423


## %%
# testing write json unite
## %%
# add Fe II lines
# tab_old = el.table[el.table['ion'] != 'Fe II']
# p = Path(dotfit.__file__).parent / 'data/emission_lines/feii_forbidden.csv'
# el.table = vstack([tab_old, Table.read(p)])
# el.table.sort('wave_vac')
# el.table.write(Path(dotfit.__file__).parent / 'data/emission_lines/emission_lines.csv')

# add He I lines
# tab_old = el.table[el.table['ion'] != 'He I']
# p = Path(dotfit.__file__).parent / 'data/emission_lines/hei.csv'
# del tab_old['note']
# el.table = Table.read(p)
# el.table = el.regenerate_table(tolerance=0.1)
# el.table = vstack([el.table, tab_old])
# el.table.sort('wave_vac')
# el.table.write(Path(dotfit.__file__).parent / 'data/emission_lines/emission_lines.csv', overwrite=True)

el.groups
group = [{'name': 'default'}]
group.append({'emission1': '[OII],[NII]'})
group.append({'emission2': '[SII]'})
group.append({'narrow': 'CaII-3935*'})
# group.append({'broad1': 'Ha,Hb,Hg', 'TieDispersion': False})
# group.append({'broad2': 'OI, HeI-7067, HeI-5877'})
unite_config = el.to_unite(group, save=True)
unite_config
# group = [{'name': 'FeII'}]
# group.append({'emission1': '[OIII]'})
# group.append({'emission2': '[FeII]-4415', 'multiplet': True})
# el.to_unite(group)

# group = [{'name': 'FeII'}]
# group.append({'emission1': '[Fe II]', 'wave': [4200, 4500]})
# group.append({'broad1': 'Ha,Hb,Hg', 'additional': ['broad', 'absorption']})
# unite_config = el.to_unite(group, save=True)

# group = [{'name': 'Hg+FeII_lorentzian'}]
# group.append({'emission1': 'Hd,Hg', 'additional': 'lorentzian'})
# group.append({'emission2': '[OIII]-4364,[SII]-4070'})
# # group.append({'emission2': '[FeII]-4289,[FeII]-4245,[FeII]-4418', 'multiplet': True})
# # group.append({'emission3': 'HeI-4027,HeI-4145,HeI-4389,HeI-4473'})
# group.append({'absorption': 'Hg,Hd'})  # do not tie dispersion of broad lines
# config = el.to_unite(group)


# ------------------------------------------------------------------ curate lines
# %%
# IRON
# forbidden iron lines
Aki_threshold = 0.01
feii_f = get_line_nist(
    ion='Fe II',
    wave=[3000, 11_500],
    tolerance=1,
    single=False,
    sortkey='wave_vac',
    threshold=f'Aki>{Aki_threshold}',
    clear_cache=False,
    verbose=True,
    classification='forbidden',
)
feii_f.sort('wave_vac')
feii_f = calculate_multiplet_emissivities(feii_f, Te=10_000)
feii_f.write(datadir / 'feii_forbidden.csv', overwrite=True)

# broad from fantasy
feii_p = Table.read(datadir / 'pool/feii_model_fantasy.csv')
feii_p['ion'] = 'Fe II'
feii_p['wave_vac'] = air_to_vacuum(feii_p['wav'])
feii_p['line_ratio'] = feii_p['Int']
feii_p.rename_column('ime', 'terms')

# permitted iron lines
# add Sigut at 3000-3700AA
# feii_p_sigut = sigut['ion', 'wave_vac', 'terms']
# feii_p_sigut['line_ratio'] = 1.0 * sigut['line_ratio']
# feii_p_sigut['terms'] = [r.split('-')[0] for r in sigut['terms']]
# for k in np.unique(feii_p_sigut['terms']):
#     idx = feii_p_sigut['terms'] == k
#     feii_p_sigut['line_ratio'][idx] = (
#         feii_p_sigut['line_ratio'][idx] / feii_p_sigut['line_ratio'][idx].max()
#     )

# feii_p_sigut = feii_p_sigut[feii_p_sigut.colnames]
# feii_p = feii_p_sigut[feii_p_sigut.colnames]

# ww = feii_p_sigut['wave_vac']
# feii_p = vstack([feii_p_sigut[(ww > 3000) & (ww < 3700)], feii_p])
# feii_p = assign_multiplets(feii_p)
# feii_p.write(datadir / 'feii_permitted.csv', overwrite=True)

# coronal
import re
from dotfit import EmissionLines

coronal = Table.read(datadir / 'pool/fantasy_coronal.csv')
coronal.rename_columns(['line', 'position'], ['ion', 'wave'])
coronal['wave_vac'] = air_to_vacuum(coronal['wave'])
coronal['ion'] = [re.sub(r'(\[?[A-Z][a-z]?)([IVX]+\]?)', r'\1 \2', ion) for ion in coronal['ion']]
el = EmissionLines()
el.table = coronal
coronal = el.regenerate_table()
coronal.write(datadir / 'coronal.csv', overwrite=True)

# sigut 03 fluorescent lines
sigut.write(datadir / 'feii_sigut03.csv', overwrite=True)


# %%
from dotfit import EmissionLines
from dotfit.emission_lines import (
    multiplet_ratios,
    assign_multiplets,
    calculate_multiplet_emissivities,
    get_line_nist,
)
from astropy import conf

conf.max_lines = 1000  # default number of rows astropy will print
conf.max_width = 200
el = EmissionLines()

# sigut = read_sigut_table(datadir / 'pool' / "sigut_03.tab")
# sigut.write(datadir / 'feii_sigut03.csv', overwrite=True)
# tab_line = apply_multiplet_rules(el.get_table('Ca II'), verbose=True)
# tab_line = apply_multiplet_rules(el.table, verbose=True)
# tab_line.write('emission_lines.csv', overwrite=True)

# %%
# --------------------------- hydrogen
import re

el = EmissionLines()
t = el.get_table('H I')
pattern = r'^(LyA|LyB+|H([1-9]|1[0-9]|2[0-9]|3[0-9]|40)|Pa[a-z]+|Pa([1-9]|1[0-9]|20))$'
tsel = t[[bool(re.match(pattern, str(key), re.IGNORECASE)) for key in t['key']]]
tsel['ion', 'wave_vac', 'key', 'configuration', 'terms', 'Ei', 'Ek', 'Aki', 'fik', 'gigk'].write(
    datadir / 'h_lines.csv', overwrite=True
)

# %%
# # ----------------------------------- Helium ---
ww = [2900, 8000]
lines = get_line_nist(
    ion='He I',
    wave=ww,
    single=False,
    tolerance=1,
    threshold='gf>0.01',
    clear_cache=False,
    verbose=True,
    classification='permitted',
)
lines = lines.group_by(["ion", "multiplet"])
lines1 = vstack([g[np.argmax(g['gf'])] if g['multiplet'][0] > 0 else g for g in lines.groups])

ww = [8000, 11_000]
lines = get_line_nist(
    ion='He I',
    wave=ww,
    single=False,
    tolerance=1,
    threshold='gf>0.1',
    clear_cache=False,
    verbose=True,
    classification='permitted',
)
lines = lines.group_by(["ion", "multiplet"])
lines = vstack([g[np.argmax(g['gf'])] if g['multiplet'][0] > 0 else g for g in lines.groups])
lines = vstack([lines1, lines])
lines.sort('wave_vac')
lines['ion', 'wave_vac', 'key', 'configuration', 'terms', 'Ei', 'Ek', 'Aki', 'fik', 'gigk'].write(
    datadir / 'hei_lines.csv', overwrite=True
)
# el.get_table('He I')['ion', 'wave_vac', 'configuration', 'key', 'terms', 'line_ratio']

# He II
# fantasy_helium.csv
el.get_table('He II')[
    'ion', 'wave_vac', 'key', 'configuration', 'terms', 'Ei', 'Ek', 'Aki', 'fik', 'gigk'
].write(datadir / 'heii_lines.csv', overwrite=True)

# lines['ion', 'wave_vac', 'configuration', 'key', 'terms'].write(datadir / 'he_lines.csv', overwrite=True)
# el.get_table('He I')['ion', 'wave_vac', 'configuration', 'key', 'terms', 'line_ratio']
# %%

# broad
# broad = Table.read(datadir / 'pool/fantasy_coronal.csv')
# coronal.rename_columns(['line', 'position'], ['ion', 'wave'])
oi = el.get_table('O I')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms']
caii = el.get_table('Ca II')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms']
ni = el.get_table('N I')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms']
nai = el.get_table('Na I')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms']
civ = el.get_table('C IV')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms']
ciii = el.get_table('C III]')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms']
mgii = el.get_table('Mg II')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms']
broad = vstack([oi, caii, ni, nai, civ, ciii, mgii])
broad.sort('wave_vac')
broad = calculate_multiplet_emissivities(assign_multiplets(broad), Te=10_000)
broad['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'Aki', 'fik', 'gigk', 'terms'].write(
    datadir / 'broad_permitted.csv', overwrite=True
)

#
tab = EmissionLines().table.to_pandas()
narrow_non_fe = tab[
    tab['ion'].str.endswith(']') & ~tab['ion'].str.contains('Fe') & ~tab['key'].isin(coronal['key'])
]
tab = calculate_multiplet_emissivities(assign_multiplets(Table.from_pandas(narrow_non_fe)))
tab.write(datadir / 'narrow.csv', overwrite=True)

# %%
# ----------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------- display stuff
# from el.table write lines excluding FeII, HI, and HeI,II
narrow = el.get_table
narrow_nir = narrow[[']' in str(n) for n in narrow['ion']]]

plot_ion_models(tab=narrow_opt, spectrum_dict=monster, spectrum_key='g235m', wave_range=[6500, 7100])
plot_ion_models(tab=narrow_nir, spectrum_dict=monster, spectrum_key='g395m', wave_range=[6500, 7500])

# forbidden
# el.get_table('[O I]')['ion', 'wave_vac', 'key', 'terms']
# el.get_table(wave=[1000,2800])
sii = el.get_table('[S II]')['ion', 'wave_vac', 'key', 'Ek', 'Ei', 'gigk', 'terms']

sii = get_line_nist(
    ion='S II',
    wave=[3500, 11_000],
    single=False,
    tolerance=1,
    threshold='Aki>0.001',
    clear_cache=False,
    verbose=True,
    classification='forbidden',
)
ww = [3000, 13_000]
ww = [9500, 10500]
ww = [3120, 3700]
plot_ion_models(
    ion='S II', wave_range=ww, spectrum_dict=monster, spectrum_key='g235m', fwhm_permitted=300.0
)
# plot_ion_models(ion='S II', wave_range=ww, spectrum_dict=monster, spectrum_key='g395m', fwhm_permitted=300.0)

siii = get_line_nist(
    ion='S III',
    wave=[3500, 11_000],
    single=False,
    tolerance=1,
    threshold='Aki>0.01',
    clear_cache=False,
    verbose=True,
    classification='forbidden',
)
# %%
# O III (no Sigut)
plot_ion_models(ion='O III', wave_range=[4500, 5500], spectrum_dict=monster, Te=10_000)

plot_ion_models(ion='Fe II', wave_range=[4500, 5500], spectrum_dict=monster, Te=10_000)

# %%
# # coronal
fef = Table.read(datadir / 'feii_forbidden.csv')
fep = Table.read(datadir / 'feii_permitted.csv')
# plot_ion_models(tab=fef, wave_range=[4500, 5500], spectrum_dict=monster,  Te=10_000)
plot_ion_models(tab=fep, wave_range=[6000, 9500], spectrum_dict=monster, spectrum_key='g395m')

# %%
cor = Table.read(datadir / 'coronal.csv')
plot_ion_models(tab=cor, wave_range=[4500, 5500], spectrum_dict=monster, spectrum_key='g395m', Te=10_000)

# %%
hei = Table.read(datadir / 'hei_lines.csv')
plot_ion_models(tab=hei, wave_range=[6500, 8000], spectrum_dict=monster, spectrum_key='g395m', Te=10_000)

# %%
# He I
plot_ion_models(
    ion='He I', wave_range=[3800, 4100], spectrum_dict=monster, Te=12_000, fwhm_permitted=500.0
)  # %%

# %%
p = get_line_nist(
    ion='Fe II',
    wave=[6000, 8000],
    tolerance=1,
    sortkey='Aki',
    threshold=f'gf>0.0001',
    verbose=True,
    classification='permitted',
    multiplet_lower_only=True,
)

# https://lweb.cfa.harvard.edu/amp/ampdata/kurucz23/sekur.html
# %%
el = EmissionLines()

t = get_line_nist(
    ion='N I', wave=[7400, 8800], tolerance=1, sortkey='wave_vac', clear_cache=False, verbose=True
)

# %%

t = get_line_nist(
    ion='Fe II', wave=4258, tolerance=1, single=True, sortkey='wave_vac', clear_cache=False, verbose=True
)

# %%
t = get_line_nist(ion='O II', wave=[3000, 11000], threshold='Aki>0.01', clear_cache=False, verbose=True)

#
# %%
# 3497.348
Nist.query(3497 << u.AA, 3498 << u.AA, linename='Fe II', wavelength_type='vacuum')

Nist.query(4257 << u.AA, 4260 << u.AA, linename='Fe II', wavelength_type='vacuum')


# get_line_nist(ion='Mg I', wave=5150, tolerance=100, single=True, clear_cache=False, verbose=True)
# new_table = el.regenerate_table(tolerance=1.0, verbose=True)

# %%
# updated_table = el.add_nist_columns(columns=['fik'], tolerance=1.0, verbose=True)


nist_result = get_line_nist(
    ion='H I', wave=[1200, 1220], tolerance=1, single=True, clear_cache=False, verbose=True
)


# %%
import warnings

warnings.filterwarnings('ignore')
from astropy.table import Table
import astropy.io.fits as pyfits
from astropy.utils.data import download_file
from pathlib import Path

# Full table
# URL_PREFIX = "https://s3.amazonaws.com/msaexp-nirspec/extractions"

version = "v4.4"  # Updated September 5, 2025.  Include all public spectra even without redshift / line fits
URL_PREFIX = "https://zenodo.org/records/15472354/files/"
table_csv = f"{URL_PREFIX}/dja_msaexp_emission_lines_{version}.csv.gz"

# table_csv = 'data/rubies.csv'
# tab = utils.read_catalog(download_file(table_url, cache=CACHE_DOWNLOADS), format='csv')

p = Path(table_csv)
if p.exists():
    tab = Table.read(str(p), format='csv')
else:
    local_path = download_file(table_csv, cache=True)
    tab = Table.read(local_path, format='csv')

# tab = Table.read(download_file(table_url, cache=True), format='csv')
# name = 'rubies-egs53-nod-v3_prism-clear_4233_42046.spec.fits'

# %%

FITS_URL = "https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}"
print(len(tab))
name = 'rubies-egs53-v4_prism-clear_4233_42046.spec.fits'
idx = [name in f for f in tab['file']]
ix = np.where(idx)[0]
tab[ix]
cache_file = download_file(FITS_URL.format(**tab[ix][0]), cache=False, show_progress=True)

# %%

# %%
# Fe II with Sigut template
# ww = [3100, 3710]
ww = [3600, 4210]
# ww = [4000, 5000]
# ww = [5000, 6500]
# ww = [4400,4630]
# ww = [5000,6000]
# ww = [6000,8380]
# ww = [8000,8500]
# ww = [8800,10100]
# plot_ion_models(ion='Fe I', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g395m',gf_threshold=0.01)
# plot_ion_models(ion='Ti II', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g395m',gf_threshold=0.01)
plot_ion_models(
    ion='Fe II',
    wave_range=ww,
    sigut_table=sigut,
    spectrum_dict=monster,
    spectrum_key='g235m',
    gf_threshold=0.001,
    merge_semiforbidden=True,
)
plot_ion_models(
    ion='Fe I',
    wave_range=ww,
    sigut_table=sigut,
    spectrum_dict=monster,
    spectrum_key='g235m',
    gf_threshold=0.001,
    merge_semiforbidden=True,
)
plot_ion_models(
    ion='Ti II',
    wave_range=ww,
    sigut_table=sigut,
    spectrum_dict=monster,
    spectrum_key='g235m',
    gf_threshold=0.001,
    merge_semiforbidden=True,
)
plot_ion_models(
    ion='Ba II',
    wave_range=ww,
    sigut_table=sigut,
    spectrum_dict=monster,
    spectrum_key='g235m',
    gf_threshold=0.001,
    merge_semiforbidden=True,
)
plot_ion_models(
    ion='Sc II',
    wave_range=ww,
    sigut_table=sigut,
    spectrum_dict=monster,
    spectrum_key='g235m',
    gf_threshold=0.001,
    merge_semiforbidden=True,
)
plot_ion_models(
    ion='Mg I',
    wave_range=ww,
    sigut_table=sigut,
    spectrum_dict=monster,
    spectrum_key='g235m',
    gf_threshold=0.001,
    merge_semiforbidden=True,
)
# plot_ion_models(ion='Fe I', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g395m',gf_threshold=0.01)
# plot_ion_models(ion='Fe II', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g235m')
# plot_ion_models(ion='Fe II', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g235m')
# plot_ion_models(ion='Ba I', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g235m',gf_threshold=0.001)
# plot_ion_models(ion='Ti II', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g235m',gf_threshold=0.001)
# plot_ion_models(ion='Sc I', wave_range=ww, sigut_table=sigut, spectrum_dict=monster, spectrum_key='g235m',gf_threshold=0.001)


# %%
# Fe II, Fe I lines
from pathlib import Path
import dotfit
import numpy as np
from dotfit.emission_lines import (
    read_kurucz_table,
    EmissionLines,
    classify_transition,
    assign_multiplets,
    calculate_multiplet_emissivities,
)
from astropy import conf

conf.max_lines = 4000  # default number of rows astropy will print
conf.max_width = 200

el = EmissionLines()

# kfile = Path(dotfit.__file__).parent / 'data/emission_lines/pool/kurucz_fei.dat'
# kfile = Path(dotfit.__file__).parent / 'data/emission_lines/pool/kurucz_feii.dat'
kfile = Path(dotfit.__file__).parent / 'data/emission_lines/pool/kurucz_TiII.dat'
tab = read_kurucz_table(str(kfile))
# idx = (tab['gf'] > 1e-5) & (tab['wave_vac'] < 11_000) & (tab['wave_vac'] > 3500)
idx = tab['gf'] > 1e-5  # & (tab['wave_vac'] < 11_000) & (tab['wave_vac'] > 1000)
mtab = calculate_multiplet_emissivities(assign_multiplets(tab[idx], lower_only=True), Te=5_000)

# %%
termlo = np.array([t.split('-')[0] for t in mtab['terms']])
it = (termlo != '') & (mtab['wave_vac'] > 1_500) & (mtab['wave_vac'] < 4_650) & (mtab['Ei'] < 4.0)
# it = (termlo != '') & (mtab['wave_vac'] > 1_200) & (mtab['wave_vac'] < 3_000)
it = (termlo != '') & (mtab['wave_vac'] > 1_200) & (mtab['wave_vac'] < 1_900)
it = (termlo != '') & (mtab['wave_vac'] > 2000) & (mtab['wave_vac'] < 3_000) & (mtab['Ei'] < 4.0)
# it = (termlo == 'b4F') & (mtab['wave_vac'] > 1_000) & (mtab['wave_vac'] < 6_500) & (mtab['Ei'] < 4.0)
# it = (termlo == 'z4D')
# it = (termlo == 'a6S') & (mtab['wave_vac'] > 4_000)
plot_lines(mtab[it], legend=True, fwhm_kms=3000)
ax = plt.gca()
ax.set_xlim(1200, 2000)
ax.set_xlim(2000, 3000)
plt.show()
len(mtab[it])
t = mtab[it]['multiplet_term', 'wave_air', 'line_ratio', 'ion', 'fik', 'Ei', 'Ek', 'gigk', 'gf']
t['line_ratio'] = t['line_ratio'] / t['line_ratio'].max()
# t
# ix = (mtab['ion'] == 'Fe II') & (mtab['multiplet'] == 32)
# plot_multiplet_gaussians_astropy(mtab[ix])
# gtab = calculate_log_gf(mtab)
# mtab[mtab['multiplet']==1000]
# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# group Ei by term (skip empty terms)
groups = defaultdict(list)
for term, ei in zip(mtab['multiplet_term'][it], mtab['Ei'][it]):
    if not term:
        continue
    groups[str(term)].append(float(ei))

# compute averages
terms = []
avg_ei = []
for term, eis in groups.items():
    terms.append(term)
    avg_ei.append(np.mean(eis))

# sort by average Ei
order = np.argsort(avg_ei)
terms_sorted = [terms[i] for i in order]
avg_sorted = [avg_ei[i] for i in order]

# plot horizontal bar (terms on y, average Ei in eV on x)
plt.figure(figsize=(8, max(4, len(terms_sorted) * 0.2)))
plt.barh(range(len(terms_sorted)), avg_sorted, color='C0')
plt.yticks(range(len(terms_sorted)), terms_sorted)
plt.xlabel('Average lower-level energy (eV)')
plt.tight_layout()
plt.show()
# %%

it = (sigut['wave_vac'] > 4000) & (sigut['wave_vac'] < 5_000)
plot_lines(sigut[it], legend=True, fwhm_kms=3000)
plt.show()
it = (sigut['multiplet_term'] == 'b4F') & (sigut['wave_vac'] > 1000) & (sigut['wave_vac'] < 3_000)
plot_lines(sigut[it], legend=True, fwhm_kms=3000)
plt.show()
it = (sigut['wave_vac'] > 2000) & (sigut['wave_vac'] < 3_000)
plot_lines(sigut[it], legend=True, fwhm_kms=3000)
ax.set_xlim(2000, 3000)
plt.show()
it = (sigut['wave_vac'] > 1200) & (sigut['wave_vac'] < 1_900)
plot_lines(sigut[it], legend=True, fwhm_kms=3000)
ax = plt.gca()
ax.set_xlim(1200, 2000)

# %%
# https://www.aanda.org/articles/aa/pdf/2015/01/aa23152-13.pdf
import numpy as np
import pyneb as pn

He2 = pn.RecAtom('He', 2)  # hydrogenic He+
ne = 1e6
Tes = np.geomspace(1e4, 1e5, 9)  # 1e4 ... 1e5 K
print("Te[K]   ratio=I(4686)/I(8237)")
for Te in Tes:
    # Option A: address lines by wavelength (vacuum Å)
    #    j4686 = He2.getEmissivity(Te, ne, wave=4686.0)
    #    j8237 = He2.getEmissivity(Te, ne, wave=8237.0)
    # If your PyNeb data table uses slightly different Ritz wavelengths, you can use labels instead:
    j4686 = He2.getEmissivity(Te, ne, label='4_3')  # 4→3
    j8237 = He2.getEmissivity(Te, ne, label='9_5')  # 9→5
    print(f"{Te:7.1f}  {j4686/j8237:8.3f}")
