// Per-model availability from /providers/validate (ModelInfo.available,
// unavailable_reason). A gateway such as OpenCode Go lists models the key may
// not call; those stay visible in pickers but cannot be chosen.

export const UNAVAILABLE_REASON_LABELS = {
  region_opt_in: 'Needs opt-in (region)',
  data_policy_opt_in: 'Needs opt-in (data policy)',
  plan: 'Not in your plan',
  unsupported: 'Not supported by the gateway',
};

// Models validated before the field existed carry no `available`; they stay selectable.
export function isModelSelectable(model) {
  return model?.available !== false;
}

export function unavailableLabel(model) {
  return UNAVAILABLE_REASON_LABELS[model?.unavailable_reason] || 'Unavailable';
}

export function firstSelectableModel(models) {
  return (models || []).find(isModelSelectable) || null;
}
