# Bepo Mobile

The native Expo/React Native client for Bepo. It connects to the private Bepo API on Railway and runs on iOS and Android.

## What it includes

- Take a photo or choose one from the phone library
- Save notes, easy `#hashtags`, optional moods, and optional GPS coordinates
- Reuse existing tags from suggestions and see saved tags as clear chips
- Pick several moods from a compact selector or add your own; used moods are suggested again
- Browse all saved memories
- Search by image/text meaning
- Ask Bepo natural-language questions about saved memories
- Type a saved place naturally or use `@place` suggestions; the removable place chip searches that whole branch
- Assign a new photo to a manual place with the same `@place` picker, independently of its GPS metadata
- Store the Railway API key in iOS Keychain or Android Keystore through Expo SecureStore

The Railway API URL is prefilled. The API key is deliberately **not** committed to this repository or embedded in the app. Paste `BEPO_API_KEY` during the one-time connection screen.

## Try it on an iPhone with Expo Go

1. Install **Expo Go** from the iPhone App Store.
2. From this `mobile` directory, run:

   ```bash
   npm install
   npx expo start --tunnel
   ```

3. Scan the QR code with the iPhone Camera app.
4. On Bepo's first screen, paste the `BEPO_API_KEY` stored in Railway.

Expo Go is the quickest testing path. It runs the React Native app inside Expo's developer shell; it is not the final App Store package.

## Build a standalone iPhone app

An Apple Developer Program membership is required for TestFlight/App Store distribution.

```bash
npm install --global eas-cli
eas login
eas build:configure
eas build --platform ios --profile production
eas submit --platform ios
```

EAS Build performs the iOS build in the cloud, so the source project can be prepared from Windows. Apple will still require the account holder to complete agreements, signing, privacy details, and any two-factor authentication prompts.

## Checks

```bash
npm run typecheck
npx expo export --platform ios
```

## App identity

- iOS bundle identifier: `com.enlorik.bepo`
- Android package: `com.enlorik.bepo`
- Default API: `https://bepo-production.up.railway.app`

