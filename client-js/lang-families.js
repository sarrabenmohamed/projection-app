// Maps FLORES-200 language codes to linguistic families, for the default
// "color by language family" mode (see point-layer.js). Unknown codes
// (e.g. from an uploaded dataset with its own language codes) fall back
// to a shared "Other" bucket rather than going uncolored.

const LANG_FAMILY = {
  eng_Latn: 'Germanic', deu_Latn: 'Germanic', nld_Latn: 'Germanic', swe_Latn: 'Germanic', dan_Latn: 'Germanic',
  spa_Latn: 'Romance', fra_Latn: 'Romance', por_Latn: 'Romance', ita_Latn: 'Romance', ron_Latn: 'Romance',
  zho_Hans: 'Sino-Tibetan', zho_Hant: 'Sino-Tibetan', mya_Mymr: 'Sino-Tibetan',
  jpn_Jpan: 'Japonic',
  kor_Hang: 'Koreanic',
  rus_Cyrl: 'Slavic', pol_Latn: 'Slavic', ukr_Cyrl: 'Slavic', ces_Latn: 'Slavic',
  tur_Latn: 'Turkic',
  vie_Latn: 'Austroasiatic', khm_Khmr: 'Austroasiatic',
  ind_Latn: 'Austronesian',
  tha_Thai: 'Kra-Dai',
  hun_Latn: 'Uralic', fin_Latn: 'Uralic',
  heb_Hebr: 'Afroasiatic', arb_Arab: 'Afroasiatic', amh_Ethi: 'Afroasiatic',
  pes_Arab: 'Indo-Iranian', urd_Arab: 'Indo-Iranian', hin_Deva: 'Indo-Iranian', ben_Beng: 'Indo-Iranian',
  guj_Gujr: 'Indo-Iranian', pan_Guru: 'Indo-Iranian', sin_Sinh: 'Indo-Iranian',
  tam_Taml: 'Dravidian', tel_Telu: 'Dravidian', mal_Mlym: 'Dravidian', kan_Knda: 'Dravidian',
};

export function familyOf(langCode) {
  return LANG_FAMILY[langCode] || 'Other';
}
