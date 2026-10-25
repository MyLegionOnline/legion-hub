// The first-run guide's shared pieces (lib/onboarding.jsx holds the provider):
// the step list, the context with its idle value and the hook components read.
import { createContext, useContext } from 'react';

export const ONBOARDING_STEPS = ['secret', 'provider', 'model', 'agents'];

const SECRETS_CHANGED = 'legion:secrets-changed';

export const SECRETS_CHANGED_EVENT = SECRETS_CHANGED;

export function notifySecretsChanged() {
  window.dispatchEvent(new Event(SECRETS_CHANGED));
}

// What components outside the provider (and a finished guide) see.
export const IDLE = {
  step: null,
  stepNumber: 0,
  total: ONBOARDING_STEPS.length,
  skip: () => {},
  finish: () => {},
  agentsWithoutModel: 0,
  applyDefaultModel: async () => {},
  applying: false,
  applyError: '',
};

export const OnboardingContext = createContext(IDLE);

export function useOnboarding() {
  return useContext(OnboardingContext);
}
