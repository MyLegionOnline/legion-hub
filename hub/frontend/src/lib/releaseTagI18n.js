// Git panel release-tag copy. The Hub UI is English; add locales here when it is translated.
const messages = {
  en: { open: 'Release tag', title: 'Mark a release', version: 'Version', message: 'Message (optional)',
    hint: 'Publish main first (Push main or publish) if this commit is not on GitHub.',
    patch: 'Next patch', minor: 'Next minor', submit: 'Create release tag', cancel: 'Cancel',
    exists: 'This version is taken; tags are never moved.', unpublished: 'Publish main first.',
    failure: 'Could not create the release tag.', suggestions: 'Could not load version suggestions.',
    success: 'Release tag created', loading: 'Loading version suggestions…' },
};
export function releaseTagMessages(locale = navigator.language) {
  return messages[locale?.toLowerCase().split('-')[0]] || messages.en;
}
