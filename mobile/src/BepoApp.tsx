import { File } from 'expo-file-system';
import Ionicons from '@expo/vector-icons/Ionicons';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import { Asset as MediaLibraryAsset } from 'expo-media-library';
import * as SecureStore from 'expo-secure-store';
import { StatusBar } from 'expo-status-bar';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  KeyboardAvoidingView,
  Linking,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  Text,
  TextInput,
  View,
} from 'react-native';
import { styles } from './styles';

const DEFAULT_API_URL = 'https://bepo-production.up.railway.app';
const KEY_STORAGE_NAME = 'bepo_api_key';
const URL_STORAGE_NAME = 'bepo_api_url';

type Screen = 'chat' | 'memories' | 'memory' | 'settings';
type MemoryContext = 'physical' | 'online' | 'mixed' | 'unknown';
type ShoppingStatus = 'want' | 'ordered' | 'bought' | 'returned' | 'no_longer_want';

type Memory = {
  id: number;
  timestamp: string;
  added_at?: string;
  taken_at?: string | null;
  taken_at_source?: 'photo' | 'camera' | 'manual' | null;
  note_created_at?: string | null;
  note_updated_at?: string | null;
  note: string | null;
  user_note?: string | null;
  bepo_summary?: string | null;
  tags?: string | null;
  mood?: string | null;
  place_hint?: string | null;
  context_type?: MemoryContext;
  shopping_status?: ShoppingStatus | null;
  shopping_status_updated_at?: string | null;
  lat: number | null;
  lon: number | null;
  location_source?: 'photo' | 'current' | 'manual' | null;
  image_url: string;
  map_url: string | null;
  score?: number;
};

type ChatResponse = { status: string; answer: string; memories: Memory[] };
type Requester = (path: string, options?: RequestInit) => Promise<any>;

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  photoUri?: string;
  tags?: string[];
  moods?: string[];
  memories?: Memory[];
  meta?: string;
  error?: boolean;
};

type Coordinates = { lat: number; lon: number };

type PendingPhoto = {
  asset: ImagePicker.ImagePickerAsset;
  source: 'camera' | 'library';
  takenAt: string | null;
  takenAtSource: 'photo' | 'camera' | null;
  coordinates: Coordinates | null;
  locationSource: 'photo' | 'current' | null;
  placeLabel: string | null;
  metadataLoading: boolean;
};

function cleanUrl(value: string) {
  return value.trim().replace(/\/+$/, '');
}

function absoluteUrl(baseUrl: string, path: string) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${cleanUrl(baseUrl)}${path.startsWith('/') ? '' : '/'}${path}`;
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() === new Date().getFullYear() ? undefined : 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

function exifValue(exif: Record<string, any> | null | undefined, keys: string[]) {
  if (!exif) return null;
  for (const key of keys) {
    if (exif[key] !== undefined && exif[key] !== null) return exif[key];
  }
  return null;
}

function rationalNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  const fraction = trimmed.match(/^(-?\d+(?:\.\d+)?)\s*\/\s*(-?\d+(?:\.\d+)?)$/);
  if (fraction) {
    const denominator = Number(fraction[2]);
    return denominator ? Number(fraction[1]) / denominator : null;
  }
  const number = Number(trimmed);
  return Number.isFinite(number) ? number : null;
}

function exifCoordinate(value: unknown, reference: unknown): number | null {
  let coordinate: number | null = null;
  if (Array.isArray(value)) {
    const parts = value.map(rationalNumber);
    if (parts.length >= 3 && parts.every((part) => part !== null)) {
      coordinate = parts[0]! + parts[1]! / 60 + parts[2]! / 3600;
    }
  } else if (typeof value === 'string' && value.includes(',')) {
    const parts = value.split(',').map((part) => rationalNumber(part));
    if (parts.length >= 3 && parts.every((part) => part !== null)) {
      coordinate = parts[0]! + parts[1]! / 60 + parts[2]! / 3600;
    }
  } else {
    coordinate = rationalNumber(value);
  }
  if (coordinate === null) return null;
  const direction = typeof reference === 'string' ? reference.toUpperCase() : '';
  return direction === 'S' || direction === 'W' ? -Math.abs(coordinate) : coordinate;
}

function coordinatesFromExif(exif: Record<string, any> | null | undefined): Coordinates | null {
  const lat = exifCoordinate(
    exifValue(exif, ['GPSLatitude', 'latitude', 'Latitude']),
    exifValue(exif, ['GPSLatitudeRef', 'latitudeRef']),
  );
  const lon = exifCoordinate(
    exifValue(exif, ['GPSLongitude', 'longitude', 'Longitude']),
    exifValue(exif, ['GPSLongitudeRef', 'longitudeRef']),
  );
  if (lat === null || lon === null || Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

function takenAtFromExif(exif: Record<string, any> | null | undefined): string | null {
  const raw = exifValue(exif, ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime', 'CreationDate']);
  if (typeof raw !== 'string') return null;
  const match = raw.trim().match(/^(\d{4})[:/-](\d{2})[:/-](\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?/);
  if (!match) return null;
  const [, year, month, day, hour = '00', minute = '00', second = '00'] = match;
  return `${year}-${month}-${day}T${hour}:${minute}:${second}`;
}

async function placeNameForCoordinates(coordinates: Coordinates): Promise<string | null> {
  try {
    const [place] = await Location.reverseGeocodeAsync({
      latitude: coordinates.lat,
      longitude: coordinates.lon,
    });
    if (!place) return null;
    const parts = [place.city || place.district || place.subregion, place.region, place.country].filter(Boolean);
    return [...new Set(parts)].slice(0, 2).join(', ') || null;
  } catch {
    return null;
  }
}

async function readLibraryPhotoHistory(asset: ImagePicker.ImagePickerAsset) {
  let takenAt = takenAtFromExif(asset.exif);
  let coordinates = coordinatesFromExif(asset.exif);

  if (asset.assetId) {
    try {
      const libraryAsset = new MediaLibraryAsset(asset.assetId);
      const [creationResult, locationResult] = await Promise.allSettled([
        libraryAsset.getCreationTime(),
        libraryAsset.getLocation(),
      ]);
      if (!takenAt && creationResult.status === 'fulfilled' && creationResult.value) {
        takenAt = new Date(creationResult.value).toISOString();
      }
      if (!coordinates && locationResult.status === 'fulfilled' && locationResult.value) {
        coordinates = { lat: locationResult.value.latitude, lon: locationResult.value.longitude };
      }
    } catch {
      // EXIF may still have supplied either value; unavailable library details are okay.
    }
  }

  return {
    takenAt,
    coordinates,
    placeLabel: coordinates ? await placeNameForCoordinates(coordinates) : null,
  };
}

function memoryHistoryText(memory: Memory) {
  const addedAt = memory.added_at || memory.timestamp;
  const noteAt = memory.note_created_at;
  if (noteAt && Math.abs(new Date(noteAt).getTime() - new Date(addedAt).getTime()) < 60_000) {
    return `Added with note ${formatDate(addedAt)}`;
  }
  return [`Added ${formatDate(addedAt)}`, noteAt ? `Note written ${formatDate(noteAt)}` : null]
    .filter(Boolean)
    .join(' · ');
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
}

function messageId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function normalizeTag(value: string) {
  return value
    .trim()
    .replace(/^#+/, '')
    .replace(/\s+/g, '-')
    .replace(/[^\p{L}\p{N}_-]/gu, '')
    .replace(/-{2,}/g, '-')
    .toLocaleLowerCase();
}

function splitStoredTags(value: string | null | undefined) {
  if (!value) return [];
  return [...new Set(value.split(',').map(normalizeTag).filter(Boolean))];
}

function knownTagsFromMemories(memories: Memory[]) {
  const usage = new Map<string, { count: number; recentIndex: number }>();
  memories.forEach((memory, memoryIndex) => {
    splitStoredTags(memory.tags).forEach((tag) => {
      const current = usage.get(tag);
      usage.set(tag, {
        count: (current?.count || 0) + 1,
        recentIndex: current?.recentIndex ?? memoryIndex,
      });
    });
  });
  return [...usage.entries()]
    .sort((left, right) => right[1].count - left[1].count || left[1].recentIndex - right[1].recentIndex)
    .map(([tag]) => tag);
}

const STARTER_MOODS = ['calm', 'cozy', 'happy', 'excited', 'nostalgic', 'safe', 'romantic', 'sad', 'anxious'];

const CONTEXT_OPTIONS: { value: MemoryContext; label: string; icon: React.ComponentProps<typeof Ionicons>['name'] }[] = [
  { value: 'physical', label: 'A place', icon: 'location-outline' },
  { value: 'online', label: 'Online', icon: 'globe-outline' },
  { value: 'mixed', label: 'Both', icon: 'git-compare-outline' },
  { value: 'unknown', label: 'Not sure', icon: 'help-circle-outline' },
];

const SHOPPING_OPTIONS: { value: ShoppingStatus; label: string }[] = [
  { value: 'want', label: 'Want' },
  { value: 'ordered', label: 'Ordered' },
  { value: 'bought', label: 'Bought' },
  { value: 'returned', label: 'Returned' },
  { value: 'no_longer_want', label: 'No longer want' },
];

function contextLabel(value: MemoryContext | undefined) {
  return CONTEXT_OPTIONS.find((option) => option.value === value)?.label || 'Not sure';
}

function shoppingLabel(value: ShoppingStatus | null | undefined) {
  return SHOPPING_OPTIONS.find((option) => option.value === value)?.label || null;
}

function normalizeMood(value: string) {
  return value
    .trim()
    .replace(/,/g, ' ')
    .replace(/\s+/g, ' ')
    .toLocaleLowerCase();
}

function splitStoredMoods(value: string | null | undefined) {
  if (!value) return [];
  return [...new Set(value.split(',').map(normalizeMood).filter(Boolean))];
}

function knownMoodsFromMemories(memories: Memory[]) {
  const usage = new Map<string, { count: number; recentIndex: number }>();
  memories.forEach((memory, memoryIndex) => {
    splitStoredMoods(memory.mood).forEach((mood) => {
      const current = usage.get(mood);
      usage.set(mood, {
        count: (current?.count || 0) + 1,
        recentIndex: current?.recentIndex ?? memoryIndex,
      });
    });
  });
  return [...usage.entries()]
    .sort((left, right) => right[1].count - left[1].count || left[1].recentIndex - right[1].recentIndex)
    .map(([mood]) => mood);
}

function activeHashtag(value: string) {
  const match = value.match(/(?:^|\s)#([\p{L}\p{N}_-]*)$/u);
  if (!match) return null;
  return {
    query: normalizeTag(match[1]),
    start: value.lastIndexOf('#'),
  };
}

function consumeCompletedHashtags(value: string) {
  const tags: string[] = [];
  const text = value.replace(/(^|\s)#([\p{L}\p{N}][\p{L}\p{N}_-]*)(?=\s)/gu, (_match, _leading, rawTag) => {
    const tag = normalizeTag(rawTag);
    if (tag) tags.push(tag);
    return '';
  });
  return { text: text.replace(/ {2,}/g, ' ').trimStart(), tags };
}

function finalizeTaggedNote(value: string, selectedTags: string[]) {
  const typedTags: string[] = [];
  const note = value
    .replace(/(^|\s)#([\p{L}\p{N}][\p{L}\p{N}_-]*)/gu, (_match, leading, rawTag) => {
      const tag = normalizeTag(rawTag);
      if (tag) typedTags.push(tag);
      return leading || '';
    })
    .replace(/ {2,}/g, ' ')
    .replace(/\s+([.,!?;:])/g, '$1')
    .trim();
  return {
    note,
    tags: [...new Set([...selectedTags, ...typedTags].map(normalizeTag).filter(Boolean))],
  };
}

export default function BepoApp() {
  const [screen, setScreen] = useState<Screen>('chat');
  const [selectedMemoryId, setSelectedMemoryId] = useState<number | null>(null);
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL);
  const [apiKey, setApiKey] = useState('');
  const [draftUrl, setDraftUrl] = useState(DEFAULT_API_URL);
  const [draftKey, setDraftKey] = useState('');
  const [hydrating, setHydrating] = useState(true);
  const [connecting, setConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const [memories, setMemories] = useState<Memory[]>([]);
  const [loadingMemories, setLoadingMemories] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const selectedMemory = memories.find((memory) => memory.id === selectedMemoryId) || null;

  const request = useCallback(
    async (path: string, options: RequestInit = {}, keyOverride?: string, urlOverride?: string) => {
      const key = keyOverride ?? apiKey;
      const base = cleanUrl(urlOverride ?? apiUrl);
      const headers = new Headers(options.headers);
      headers.set('X-API-Key', key);
      const response = await fetch(absoluteUrl(base, path), { ...options, headers });
      if (!response.ok) {
        let message = `Bepo returned ${response.status}.`;
        try {
          const body = await response.json();
          if (body?.detail) message = String(body.detail);
        } catch {
          // Keep the status-based fallback for non-JSON responses.
        }
        if (response.status === 401) {
          message = 'That API key was not accepted. Copy BEPO_API_KEY from Railway and try again.';
        }
        throw new Error(message);
      }
      if (response.status === 204) return null;
      return response.json();
    },
    [apiKey, apiUrl],
  );

  const loadMemories = useCallback(
    async (quiet = false) => {
      if (!apiKey) return;
      if (!quiet) setLoadingMemories(true);
      try {
        setMemories((await request('/memories')) as Memory[]);
      } catch (error) {
        const message = errorMessage(error);
        if (quiet) {
          setConnectionError(message);
          setScreen('settings');
        } else {
          Alert.alert('Could not load memories', message);
        }
      } finally {
        setLoadingMemories(false);
        setRefreshing(false);
      }
    },
    [apiKey, request],
  );

  useEffect(() => {
    (async () => {
      try {
        const [storedKey, storedUrl] = await Promise.all([
          SecureStore.getItemAsync(KEY_STORAGE_NAME),
          SecureStore.getItemAsync(URL_STORAGE_NAME),
        ]);
        const resolvedUrl = storedUrl || DEFAULT_API_URL;
        setApiUrl(resolvedUrl);
        setDraftUrl(resolvedUrl);
        if (storedKey) {
          setApiKey(storedKey);
          setDraftKey(storedKey);
        }
      } finally {
        setHydrating(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (apiKey && !hydrating) loadMemories(true);
  }, [apiKey, hydrating, loadMemories]);

  async function saveConnection() {
    const nextUrl = cleanUrl(draftUrl);
    const nextKey = draftKey.trim();
    if (!nextUrl.startsWith('https://')) {
      setConnectionError('Use an HTTPS address for Bepo.');
      return;
    }
    if (!nextKey) {
      setConnectionError('Paste your BEPO_API_KEY to continue.');
      return;
    }
    setConnecting(true);
    setConnectionError('');
    try {
      await request('/memories', {}, nextKey, nextUrl);
      await Promise.all([
        SecureStore.setItemAsync(KEY_STORAGE_NAME, nextKey),
        SecureStore.setItemAsync(URL_STORAGE_NAME, nextUrl),
      ]);
      setApiUrl(nextUrl);
      setApiKey(nextKey);
      setDraftUrl(nextUrl);
      setDraftKey(nextKey);
      setScreen('chat');
    } catch (error) {
      setConnectionError(errorMessage(error));
    } finally {
      setConnecting(false);
    }
  }

  function openMemory(memoryId: number) {
    setSelectedMemoryId(memoryId);
    setScreen('memory');
  }

  function updateMemory(updatedMemory: Memory) {
    setMemories((current) => current.map((memory) => (
      memory.id === updatedMemory.id ? updatedMemory : memory
    )));
  }

  if (hydrating) {
    return (
      <View style={styles.centeredPage}>
        <StatusBar style="dark" />
        <BrandMark size={72} />
        <ActivityIndicator color="#262624" style={styles.loadingMark} />
      </View>
    );
  }

  if (!apiKey) {
    return (
      <KeyboardAvoidingView style={styles.setupPage} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <StatusBar style="dark" />
        <ScrollView contentContainerStyle={styles.setupContent} keyboardShouldPersistTaps="handled">
          <BrandMark size={72} />
          <Text style={styles.setupEyebrow}>YOUR PRIVATE MEMORY COMPANION</Text>
          <Text style={styles.setupTitle}>Meet Bepo.</Text>
          <Text style={styles.setupCopy}>
            Send Bepo a photo to remember it, or ask a question about the moments you have saved.
          </Text>
          <ConnectionForm
            url={draftUrl}
            setUrl={setDraftUrl}
            apiKey={draftKey}
            setApiKey={setDraftKey}
            error={connectionError}
            loading={connecting}
            onSave={saveConnection}
          />
        </ScrollView>
      </KeyboardAvoidingView>
    );
  }

  return (
    <View style={styles.app}>
      <StatusBar style="dark" />
      <View style={styles.safeTop} />
      {screen === 'chat' ? (
        <ChatScreen
          request={request}
          apiUrl={apiUrl}
          apiKey={apiKey}
          memoryCount={memories.length}
          knownTags={knownTagsFromMemories(memories)}
          knownMoods={knownMoodsFromMemories(memories)}
          onMemorySaved={() => loadMemories(true)}
          onOpenMemories={() => setScreen('memories')}
          onOpenSettings={() => setScreen('settings')}
        />
      ) : null}
      {screen === 'memories' ? (
        <MemoriesScreen
          memories={memories}
          apiUrl={apiUrl}
          apiKey={apiKey}
          loading={loadingMemories}
          refreshing={refreshing}
          onRefresh={() => {
            setRefreshing(true);
            loadMemories(true);
          }}
          onBack={() => setScreen('chat')}
          onOpenMemory={openMemory}
        />
      ) : null}
      {screen === 'memory' && selectedMemory ? (
        <MemoryDetailScreen
          memory={selectedMemory}
          apiUrl={apiUrl}
          apiKey={apiKey}
          request={request}
          knownTags={knownTagsFromMemories(memories)}
          knownMoods={knownMoodsFromMemories(memories)}
          onBack={() => setScreen('memories')}
          onSaved={updateMemory}
        />
      ) : null}
      {screen === 'settings' ? (
        <SettingsScreen
          url={draftUrl}
          setUrl={setDraftUrl}
          apiKey={draftKey}
          setApiKey={setDraftKey}
          error={connectionError}
          loading={connecting}
          onSave={saveConnection}
          onBack={() => setScreen('chat')}
        />
      ) : null}
    </View>
  );
}

function ChatScreen({
  request,
  apiUrl,
  apiKey,
  memoryCount,
  knownTags,
  knownMoods,
  onMemorySaved,
  onOpenMemories,
  onOpenSettings,
}: {
  request: Requester;
  apiUrl: string;
  apiKey: string;
  memoryCount: number;
  knownTags: string[];
  knownMoods: string[];
  onMemorySaved: () => Promise<void>;
  onOpenMemories: () => void;
  onOpenSettings: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [pendingPhoto, setPendingPhoto] = useState<PendingPhoto | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [selectedMoods, setSelectedMoods] = useState<string[]>([]);
  const [moodPickerOpen, setMoodPickerOpen] = useState(false);
  const [addingMood, setAddingMood] = useState(false);
  const [moodDraft, setMoodDraft] = useState('');
  const [sending, setSending] = useState(false);
  const listRef = useRef<FlatList<ChatMessage>>(null);
  const inputRef = useRef<TextInput>(null);
  const moodInputRef = useRef<TextInput>(null);

  function addMessage(message: ChatMessage) {
    setMessages((current) => [...current, message]);
  }

  function updateMessage(id: string, patch: Partial<ChatMessage>) {
    setMessages((current) => current.map((message) => (message.id === id ? { ...message, ...patch } : message)));
  }

  function addTags(tags: string[]) {
    setSelectedTags((current) => [...new Set([...current, ...tags].map(normalizeTag).filter(Boolean))]);
  }

  function handleDraftChange(value: string) {
    if (!pendingPhoto) {
      setDraft(value);
      return;
    }
    const consumed = consumeCompletedHashtags(value);
    setDraft(consumed.text);
    if (consumed.tags.length) addTags(consumed.tags);
  }

  function selectTag(tag: string) {
    const active = activeHashtag(draft);
    if (active) setDraft(draft.slice(0, active.start).trimEnd());
    addTags([tag]);
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  function removeTag(tag: string) {
    setSelectedTags((current) => current.filter((item) => item !== tag));
  }

  function toggleMood(mood: string) {
    const normalized = normalizeMood(mood);
    if (!normalized) return;
    setSelectedMoods((current) => current.includes(normalized)
      ? current.filter((item) => item !== normalized)
      : [...current, normalized]);
  }

  function saveCustomMood() {
    const mood = normalizeMood(moodDraft);
    if (!mood) return;
    setSelectedMoods((current) => current.includes(mood) ? current : [...current, mood]);
    setMoodDraft('');
    setAddingMood(false);
  }

  async function choosePhoto(source: 'camera' | 'library') {
    try {
      const permission = source === 'camera'
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Permission needed', `Allow ${source === 'camera' ? 'camera' : 'photo'} access to share a memory with Bepo.`);
        return;
      }
      const result = source === 'camera'
        ? await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.85, exif: true })
        : await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85, exif: true });
      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        if (source === 'camera') {
          setPendingPhoto({
            asset,
            source,
            takenAt: new Date().toISOString(),
            takenAtSource: 'camera',
            coordinates: null,
            locationSource: null,
            placeLabel: null,
            metadataLoading: false,
          });
        } else {
          setPendingPhoto({
            asset,
            source,
            takenAt: null,
            takenAtSource: null,
            coordinates: null,
            locationSource: null,
            placeLabel: null,
            metadataLoading: true,
          });
          const history = await readLibraryPhotoHistory(asset);
          setPendingPhoto((current) => current?.asset.uri === asset.uri ? {
            ...current,
            ...history,
            takenAtSource: history.takenAt ? 'photo' : null,
            locationSource: history.coordinates ? 'photo' : null,
            metadataLoading: false,
          } : current);
        }
      }
    } catch (error) {
      Alert.alert('Could not open photos', errorMessage(error));
    }
  }

  async function getAutomaticLocation(): Promise<Coordinates | null> {
    try {
      let permission = await Location.getForegroundPermissionsAsync();
      if (!permission.granted && permission.canAskAgain) {
        permission = await Location.requestForegroundPermissionsAsync();
      }
      if (!permission.granted) return null;
      const cached = await Location.getLastKnownPositionAsync({
        maxAge: 5 * 60 * 1000,
        requiredAccuracy: 1000,
      });
      const location = cached ?? await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced,
      });
      return { lat: location.coords.latitude, lon: location.coords.longitude };
    } catch {
      return null;
    }
  }

  async function sendMessage(textOverride?: string) {
    const photo = pendingPhoto;
    const rawText = (textOverride ?? draft).trim();
    const taggedNote = photo ? finalizeTaggedNote(rawText, selectedTags) : { note: rawText, tags: [] };
    const text = taggedNote.note;
    const tags = taggedNote.tags;
    const moods = photo ? selectedMoods : [];
    if (sending || photo?.metadataLoading || (!text && !photo)) return;

    const userId = messageId('user');
    addMessage({
      id: userId,
      role: 'user',
      text: text || 'Remember this.',
      photoUri: photo?.asset.uri,
      tags: tags.length ? tags : undefined,
      moods: moods.length ? moods : undefined,
      meta: photo ? 'Saving memory…' : undefined,
    });
    if (textOverride === undefined) setDraft('');
    setPendingPhoto(null);
    setSelectedTags([]);
    setSelectedMoods([]);
    setMoodPickerOpen(false);
    setAddingMood(false);
    setMoodDraft('');
    setSending(true);

    try {
      if (photo) {
        let coordinates = photo.coordinates;
        let locationSource = photo.locationSource;
        if (photo.source === 'camera' && !coordinates) {
          coordinates = await getAutomaticLocation();
          locationSource = coordinates ? 'current' : null;
        }
        const form = new FormData();
        const photoFile = new File(photo.asset.uri);
        form.append('photo', photoFile, photo.asset.fileName || photoFile.name || `bepo-${Date.now()}.jpg`);
        if (text) form.append('note', text);
        if (tags.length) form.append('tags', tags.join(','));
        if (moods.length) form.append('mood', moods.join(','));
        if (photo.takenAt) {
          form.append('taken_at', photo.takenAt);
          form.append('taken_at_source', photo.takenAtSource || 'photo');
        }
        if (coordinates) {
          form.append('lat', String(coordinates.lat));
          form.append('lon', String(coordinates.lon));
          if (locationSource) form.append('location_source', locationSource);
        }
        await request('/memory', { method: 'POST', body: form });
        updateMessage(userId, {
          meta: photo.source === 'library'
            ? `Saved${photo.takenAt ? ' with original date' : ''}${coordinates ? ' and location' : ''}`
            : coordinates ? 'Saved with location' : 'Saved',
        });
        addMessage({
          id: messageId('bepo'),
          role: 'assistant',
          text: photo.source === 'library'
            ? photo.takenAt || coordinates
              ? `It’s safe with me. I kept the photo’s ${photo.takenAt ? 'original date' : ''}${photo.takenAt && coordinates ? ' and ' : ''}${coordinates ? 'location' : ''}, and separately recorded when you added this note.`
              : 'It’s safe with me. This photo did not include its original date or place, so I left those unknown and recorded when you added it.'
            : coordinates
              ? 'It’s safe with me. I saved the moment, the time, and where you were.'
              : 'It’s safe with me. I saved the moment and the time. Location wasn’t available this time.',
        });
        await onMemorySaved();
      } else {
        const response = (await request('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, top_k: 3 }),
        })) as ChatResponse;
        addMessage({
          id: messageId('bepo'),
          role: 'assistant',
          text: response.answer,
          memories: response.memories || [],
        });
      }
    } catch (error) {
      if (photo) updateMessage(userId, { meta: 'Couldn’t save' });
      addMessage({
        id: messageId('error'),
        role: 'assistant',
        text: errorMessage(error),
        error: true,
      });
    } finally {
      setSending(false);
    }
  }

  const canSend = Boolean(draft.trim() || pendingPhoto) && !sending && !pendingPhoto?.metadataLoading;
  const activeTag = pendingPhoto ? activeHashtag(draft) : null;
  const matchingTags = activeTag
    ? knownTags
      .filter((tag) => !selectedTags.includes(tag) && (!activeTag.query || tag.startsWith(activeTag.query)))
      .slice(0, 6)
    : [];
  const canCreateTag = Boolean(
    activeTag?.query
    && !selectedTags.includes(activeTag.query)
    && !knownTags.includes(activeTag.query),
  );
  const moodOptions = [...new Set([...knownMoods, ...STARTER_MOODS, ...selectedMoods])];

  return (
    <View style={styles.fill}>
      <ChatControls memoryCount={memoryCount} onOpenMemories={onOpenMemories} onOpenSettings={onOpenSettings} />
      <KeyboardAvoidingView
        style={styles.fill}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 38 : 0}
      >
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => <ChatBubble message={item} apiUrl={apiUrl} apiKey={apiKey} />}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={[styles.chatListContent, messages.length === 0 && styles.chatListEmpty]}
          ListEmptyComponent={<EmptyConversation onPrompt={(prompt) => sendMessage(prompt)} />}
          ListFooterComponent={sending ? <TypingBubble /> : <View style={styles.chatListFooter} />}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
        />
        <View style={styles.composerArea}>
          {pendingPhoto ? (
            <View style={styles.attachmentPreview}>
              <Image source={{ uri: pendingPhoto.asset.uri }} style={styles.attachmentImage} />
              <View style={styles.attachmentCopy}>
                <Text style={styles.attachmentTitle}>
                  {pendingPhoto.metadataLoading
                    ? 'Reading photo history…'
                    : pendingPhoto.takenAt
                      ? `Taken ${formatDate(pendingPhoto.takenAt)}`
                      : pendingPhoto.source === 'camera' ? 'New memory' : 'Original date unavailable'}
                </Text>
                <Text style={styles.attachmentMeta}>
                  {pendingPhoto.metadataLoading
                    ? 'Checking its original date and place'
                    : pendingPhoto.source === 'camera'
                      ? 'Current location will be added when sent'
                      : pendingPhoto.coordinates
                        ? pendingPhoto.placeLabel || 'Original photo location found'
                        : 'No saved location on this photo'}
                </Text>
                {!pendingPhoto.metadataLoading ? (
                  <Text style={styles.attachmentHistory}>Your note time will be saved separately</Text>
                ) : null}
              </View>
              <Pressable
                accessibilityLabel="Remove attached photo"
                style={styles.removeAttachment}
                onPress={() => {
                  setPendingPhoto(null);
                  setSelectedTags([]);
                  setSelectedMoods([]);
                  setMoodPickerOpen(false);
                  setAddingMood(false);
                  setMoodDraft('');
                }}
              >
                <Text style={styles.removeAttachmentText}>×</Text>
              </Pressable>
            </View>
          ) : null}
          {pendingPhoto ? (
            <View style={styles.moodControlRow}>
              {selectedMoods.length ? (
                <ScrollView
                  style={styles.selectedMoodScroller}
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  keyboardShouldPersistTaps="always"
                  contentContainerStyle={styles.selectedMoodRow}
                >
                  {selectedMoods.map((mood) => (
                    <Pressable
                      key={mood}
                      accessibilityLabel={`Remove mood ${mood}`}
                      style={styles.selectedMoodChip}
                      onPress={() => toggleMood(mood)}
                    >
                      <Text style={styles.selectedMoodText}>{mood}</Text>
                      <Ionicons name="close" size={13} color="#4F6B5B" />
                    </Pressable>
                  ))}
                </ScrollView>
              ) : <View style={styles.moodControlSpacer} />}
              <Pressable
                accessibilityLabel={moodPickerOpen ? 'Close mood choices' : 'Choose moods'}
                style={[styles.moodButton, (moodPickerOpen || selectedMoods.length > 0) && styles.moodButtonActive]}
                onPress={() => {
                  setMoodPickerOpen((current) => !current);
                  if (moodPickerOpen) setAddingMood(false);
                }}
              >
                <Ionicons name={moodPickerOpen ? 'happy' : 'happy-outline'} size={18} color="#4F6B5B" />
                {selectedMoods.length ? <Text style={styles.moodCount}>{selectedMoods.length}</Text> : null}
              </Pressable>
            </View>
          ) : null}
          {pendingPhoto && moodPickerOpen ? (
            <View style={styles.moodPicker}>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                keyboardShouldPersistTaps="always"
                contentContainerStyle={styles.moodOptionRow}
              >
                {moodOptions.map((mood) => {
                  const selected = selectedMoods.includes(mood);
                  return (
                    <Pressable
                      key={mood}
                      accessibilityLabel={`${selected ? 'Remove' : 'Choose'} mood ${mood}`}
                      style={[styles.moodOptionChip, selected && styles.moodOptionChipSelected]}
                      onPress={() => toggleMood(mood)}
                    >
                      {selected ? <Ionicons name="checkmark" size={13} color="#4F6B5B" /> : null}
                      <Text style={[styles.moodOptionText, selected && styles.moodOptionTextSelected]}>{mood}</Text>
                    </Pressable>
                  );
                })}
                <Pressable
                  accessibilityLabel="Add a custom mood"
                  style={[styles.moodOptionChip, styles.addMoodChip]}
                  onPress={() => {
                    setAddingMood(true);
                    requestAnimationFrame(() => moodInputRef.current?.focus());
                  }}
                >
                  <Ionicons name="add" size={15} color="#4F6B5B" />
                  <Text style={styles.moodOptionTextSelected}>add</Text>
                </Pressable>
              </ScrollView>
              {addingMood ? (
                <View style={styles.customMoodRow}>
                  <TextInput
                    ref={moodInputRef}
                    style={styles.customMoodInput}
                    value={moodDraft}
                    onChangeText={setMoodDraft}
                    placeholder="Name a mood…"
                    placeholderTextColor="#8B8B85"
                    returnKeyType="done"
                    onSubmitEditing={saveCustomMood}
                  />
                  <Pressable
                    accessibilityLabel="Save custom mood"
                    style={[styles.customMoodSave, !normalizeMood(moodDraft) && styles.customMoodSaveDisabled]}
                    onPress={saveCustomMood}
                    disabled={!normalizeMood(moodDraft)}
                  >
                    <Ionicons name="checkmark" size={18} color="#FFFFFF" />
                  </Pressable>
                </View>
              ) : null}
            </View>
          ) : null}
          {pendingPhoto && selectedTags.length ? (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              keyboardShouldPersistTaps="always"
              contentContainerStyle={styles.selectedTagRow}
            >
              {selectedTags.map((tag) => (
                <Pressable
                  key={tag}
                  accessibilityLabel={`Remove tag ${tag}`}
                  style={styles.selectedTagChip}
                  onPress={() => removeTag(tag)}
                >
                  <Text style={styles.selectedTagText}>#{tag}</Text>
                  <Ionicons name="close" size={13} color="#765D45" />
                </Pressable>
              ))}
            </ScrollView>
          ) : null}
          {pendingPhoto && activeTag ? (
            <View style={styles.tagSuggestions}>
              {matchingTags.length || canCreateTag ? (
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  keyboardShouldPersistTaps="always"
                  contentContainerStyle={styles.tagSuggestionRow}
                >
                  {matchingTags.map((tag) => (
                    <Pressable
                      key={tag}
                      accessibilityLabel={`Use tag ${tag}`}
                      style={styles.tagSuggestionChip}
                      onPress={() => selectTag(tag)}
                    >
                      <Text style={styles.tagSuggestionText}>#{tag}</Text>
                    </Pressable>
                  ))}
                  {canCreateTag && activeTag.query ? (
                    <Pressable
                      accessibilityLabel={`Create tag ${activeTag.query}`}
                      style={[styles.tagSuggestionChip, styles.createTagChip]}
                      onPress={() => selectTag(activeTag.query)}
                    >
                      <Ionicons name="add" size={15} color="#4F6B5B" />
                      <Text style={styles.createTagText}>#{activeTag.query}</Text>
                    </Pressable>
                  ) : null}
                </ScrollView>
              ) : (
                <Text style={styles.tagSuggestionHint}>Keep typing to make a new tag</Text>
              )}
            </View>
          ) : null}
          <View style={styles.composer}>
            <View style={styles.composerTools}>
              <Pressable accessibilityLabel="Take a photo" style={styles.composerToolButton} onPress={() => choosePhoto('camera')}>
                <Ionicons name="camera-outline" size={22} color="#393936" />
              </Pressable>
              <Pressable accessibilityLabel="Choose from photos" style={styles.composerToolButton} onPress={() => choosePhoto('library')}>
                <Ionicons name="images-outline" size={21} color="#393936" />
              </Pressable>
            </View>
            <TextInput
              ref={inputRef}
              style={styles.composerInput}
              value={draft}
              onChangeText={handleDraftChange}
              multiline
              placeholder={pendingPhoto ? 'Add a note… use # for tags' : 'Message Bepo…'}
              placeholderTextColor="#8B8B85"
            />
            <Pressable
              accessibilityLabel="Send message"
              style={[styles.sendButton, !canSend && styles.sendButtonDisabled]}
              onPress={() => sendMessage()}
              disabled={!canSend}
            >
              <Ionicons name="arrow-up" size={21} color="#FFFFFF" />
            </Pressable>
          </View>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

function ChatControls({ memoryCount, onOpenMemories, onOpenSettings }: {
  memoryCount: number;
  onOpenMemories: () => void;
  onOpenSettings: () => void;
}) {
  return (
    <View style={styles.chatControls}>
      <View style={styles.headerActions}>
        <Pressable accessibilityLabel="Open memories" style={styles.headerButton} onPress={onOpenMemories}>
          <Ionicons name="albums-outline" size={22} color="#393936" />
          {memoryCount ? <View style={styles.memoryBadge}><Text style={styles.memoryBadgeText}>{memoryCount > 99 ? '99+' : memoryCount}</Text></View> : null}
        </Pressable>
        <Pressable accessibilityLabel="Open settings" style={styles.headerButton} onPress={onOpenSettings}>
          <Ionicons name="ellipsis-horizontal" size={23} color="#393936" />
        </Pressable>
      </View>
    </View>
  );
}

function EmptyConversation({ onPrompt }: { onPrompt: (prompt: string) => void }) {
  const prompts = ['What do you remember most recently?', 'Where was one of my saved moments?'];
  return (
    <View style={styles.emptyConversation}>
      <BrandMark size={82} />
      <Text style={styles.emptyConversationTitle}>What can I remember for you?</Text>
      <Text style={styles.emptyConversationCopy}>
        Send me a photo and a few words, or ask naturally about something you have saved.
      </Text>
      <View style={styles.promptList}>
        {prompts.map((prompt) => (
          <Pressable key={prompt} style={styles.promptChip} onPress={() => onPrompt(prompt)}>
            <Text style={styles.promptChipText}>{prompt}</Text>
            <Text style={styles.promptArrow}>→</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

function ChatBubble({ message, apiUrl, apiKey }: { message: ChatMessage; apiUrl: string; apiKey: string }) {
  if (message.role === 'user') {
    return (
      <View style={styles.userMessageRow}>
        <View style={styles.userBubble}>
          {message.photoUri ? <Image source={{ uri: message.photoUri }} style={styles.userPhoto} /> : null}
          <Text style={styles.userBubbleText}>{message.text}</Text>
          {message.tags?.length ? (
            <View style={styles.userTagRow}>
              {message.tags.map((tag) => <Text key={tag} style={styles.userTag}>#{tag}</Text>)}
            </View>
          ) : null}
          {message.moods?.length ? (
            <View style={styles.userMoodRow}>
              {message.moods.map((mood) => <Text key={mood} style={styles.userMood}>{mood}</Text>)}
            </View>
          ) : null}
          {message.meta ? <Text style={styles.userBubbleMeta}>{message.meta}</Text> : null}
        </View>
      </View>
    );
  }
  return (
    <View style={styles.assistantRow}>
      <BrandMark size={30} />
      <View style={styles.assistantContent}>
        <Text style={[styles.assistantText, message.error && styles.assistantError]}>{message.text}</Text>
        {message.memories?.map((memory) => (
          <MemoryCard key={memory.id} memory={memory} apiUrl={apiUrl} apiKey={apiKey} compact />
        ))}
      </View>
    </View>
  );
}

function TypingBubble() {
  return (
    <View style={styles.assistantRow}>
      <BrandMark size={30} />
      <View style={styles.typingPill}>
        <View style={styles.typingDot} />
        <View style={styles.typingDot} />
        <View style={styles.typingDot} />
      </View>
    </View>
  );
}

function MemoriesScreen({ memories, apiUrl, apiKey, loading, refreshing, onRefresh, onBack, onOpenMemory }: {
  memories: Memory[];
  apiUrl: string;
  apiKey: string;
  loading: boolean;
  refreshing: boolean;
  onRefresh: () => void;
  onBack: () => void;
  onOpenMemory: (memoryId: number) => void;
}) {
  return (
    <View style={styles.fill}>
      <PageHeader title="Memories" subtitle={`${memories.length} saved`} onBack={onBack} />
      <FlatList
        data={memories}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={[styles.memoryList, memories.length === 0 && styles.memoryListEmpty]}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#262624" />}
        ListEmptyComponent={loading ? <ActivityIndicator color="#262624" /> : (
          <View style={styles.galleryEmpty}>
            <BrandMark size={64} />
            <Text style={styles.galleryEmptyTitle}>No memories yet</Text>
            <Text style={styles.galleryEmptyCopy}>Go back to the conversation and attach your first photo.</Text>
          </View>
        )}
        renderItem={({ item }) => (
          <MemoryCard
            memory={item}
            apiUrl={apiUrl}
            apiKey={apiKey}
            onPress={() => onOpenMemory(item.id)}
          />
        )}
      />
    </View>
  );
}

function MemoryCard({ memory, apiUrl, apiKey, compact = false, onPress }: {
  memory: Memory;
  apiUrl: string;
  apiKey: string;
  compact?: boolean;
  onPress?: () => void;
}) {
  const description = memory.user_note || memory.bepo_summary || memory.note || 'A saved moment';
  const status = shoppingLabel(memory.shopping_status);
  return (
    <Pressable
      disabled={!onPress}
      onPress={onPress}
      accessibilityRole={onPress ? 'button' : undefined}
      accessibilityLabel={onPress ? `Open memory: ${description}` : undefined}
      style={({ pressed }) => [
        styles.memoryCard,
        compact && styles.inlineMemoryCard,
        pressed && styles.memoryCardPressed,
      ]}
    >
      <Image
        source={{ uri: absoluteUrl(apiUrl, memory.image_url), headers: { 'X-API-Key': apiKey } }}
        style={[styles.memoryImage, compact && styles.inlineMemoryImage]}
        resizeMode="cover"
      />
      <View style={styles.memoryBody}>
        <View style={styles.memoryMetaRow}>
          <Text style={styles.memoryDate}>
            {memory.taken_at ? `Taken ${formatDate(memory.taken_at)}` : 'Event date unavailable'}
          </Text>
          {memory.mood ? (
            <View style={styles.memoryMoodRow}>
              {splitStoredMoods(memory.mood).map((mood) => <Text key={mood} style={styles.pill}>{mood}</Text>)}
            </View>
          ) : null}
        </View>
        <Text style={[styles.memoryDescription, compact && styles.inlineMemoryDescription]}>{description}</Text>
        {memory.tags ? (
          <View style={styles.memoryTags}>
            {splitStoredTags(memory.tags).map((tag) => <Text key={tag} style={styles.memoryTag}>#{tag}</Text>)}
          </View>
        ) : null}
        <View style={styles.memoryDetailsRow}>
          <Text style={styles.memoryContext}>{memory.context_type === 'online' ? '◎ Online' : memory.context_type === 'mixed' ? '⌖ + ◎ Both' : memory.context_type === 'physical' ? '⌖ A place' : '○ Not sorted yet'}</Text>
          {status ? <Text style={styles.memoryStatus}>{status}</Text> : null}
        </View>
        {memory.place_hint ? <Text style={styles.memoryPlace}>⌖ {memory.place_hint}</Text> : null}
        <Text style={styles.memoryHistory}>{memoryHistoryText(memory)}</Text>
        {memory.map_url ? (
          <Pressable onPress={() => Linking.openURL(memory.map_url!)}>
            <Text style={styles.mapLink}>View where this was taken ↗</Text>
          </Pressable>
        ) : null}
        {onPress ? <Text style={styles.editMemoryHint}>Open & edit  ›</Text> : null}
      </View>
    </Pressable>
  );
}

function MemoryDetailScreen({
  memory,
  apiUrl,
  apiKey,
  request,
  knownTags,
  knownMoods,
  onBack,
  onSaved,
}: {
  memory: Memory;
  apiUrl: string;
  apiKey: string;
  request: Requester;
  knownTags: string[];
  knownMoods: string[];
  onBack: () => void;
  onSaved: (memory: Memory) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [note, setNote] = useState(memory.user_note || memory.note || memory.bepo_summary || '');
  const [tags, setTags] = useState(splitStoredTags(memory.tags));
  const [moods, setMoods] = useState(splitStoredMoods(memory.mood));
  const [contextType, setContextType] = useState<MemoryContext>(memory.context_type || 'unknown');
  const [shoppingStatus, setShoppingStatus] = useState<ShoppingStatus | null>(memory.shopping_status || null);
  const [tagDraft, setTagDraft] = useState('');
  const [moodDraft, setMoodDraft] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');

  useEffect(() => {
    setEditing(false);
    setNote(memory.user_note || memory.note || memory.bepo_summary || '');
    setTags(splitStoredTags(memory.tags));
    setMoods(splitStoredMoods(memory.mood));
    setContextType(memory.context_type || 'unknown');
    setShoppingStatus(memory.shopping_status || null);
    setTagDraft('');
    setMoodDraft('');
    setSaveError('');
  }, [memory.id]);

  const description = memory.user_note || memory.bepo_summary || memory.note || 'A saved moment';
  const usedTagSuggestions = knownTags.filter((tag) => !tags.includes(tag)).slice(0, 10);
  const moodChoices = [...new Set([...moods, ...knownMoods, ...STARTER_MOODS])];

  function addTag(value = tagDraft) {
    const tag = normalizeTag(value);
    if (!tag) return;
    setTags((current) => [...new Set([...current, tag])]);
    setTagDraft('');
  }

  function addMood(value = moodDraft) {
    const mood = normalizeMood(value);
    if (!mood) return;
    setMoods((current) => [...new Set([...current, mood])]);
    setMoodDraft('');
  }

  async function saveMemory() {
    setSaving(true);
    setSaveError('');
    try {
      const updated = await request(`/memory/${memory.id}/metadata`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_note: note.trim() || null,
          tags: tags.length ? tags.join(',') : null,
          mood: moods.length ? moods.join(',') : null,
          context_type: contextType,
          shopping_status: shoppingStatus,
        }),
      }) as Memory;
      onSaved(updated);
      setNote(updated.user_note || updated.note || updated.bepo_summary || '');
      setTags(splitStoredTags(updated.tags));
      setMoods(splitStoredMoods(updated.mood));
      setContextType(updated.context_type || 'unknown');
      setShoppingStatus(updated.shopping_status || null);
      setEditing(false);
    } catch (error) {
      setSaveError(errorMessage(error));
    } finally {
      setSaving(false);
    }
  }

  return (
    <View style={styles.fill}>
      <PageHeader
        title="Memory"
        subtitle={memory.taken_at ? formatDate(memory.taken_at) : 'Saved moment'}
        onBack={onBack}
      />
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView
          contentContainerStyle={styles.detailContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <Image
            source={{ uri: absoluteUrl(apiUrl, memory.image_url), headers: { 'X-API-Key': apiKey } }}
            style={styles.detailImage}
            resizeMode="cover"
          />

          {!editing ? (
            <View style={styles.detailBody}>
              <Text style={styles.detailDescription}>{description}</Text>
              <View style={styles.detailBadgeRow}>
                <Text style={styles.contextBadge}>{contextLabel(memory.context_type)}</Text>
                {shoppingLabel(memory.shopping_status) ? (
                  <Text style={styles.statusBadge}>{shoppingLabel(memory.shopping_status)}</Text>
                ) : null}
              </View>
              {memory.tags ? (
                <View style={styles.memoryTags}>
                  {splitStoredTags(memory.tags).map((tag) => <Text key={tag} style={styles.memoryTag}>#{tag}</Text>)}
                </View>
              ) : null}
              {memory.mood ? (
                <View style={styles.detailMoodRow}>
                  {splitStoredMoods(memory.mood).map((mood) => <Text key={mood} style={styles.pill}>{mood}</Text>)}
                </View>
              ) : null}
              {memory.place_hint ? <Text style={styles.memoryPlace}>⌖ {memory.place_hint}</Text> : null}
              <Text style={styles.detailHistory}>{memoryHistoryText(memory)}</Text>
              {memory.shopping_status_updated_at ? (
                <Text style={styles.detailHistory}>Shopping stage changed {formatDate(memory.shopping_status_updated_at)}</Text>
              ) : null}
              {memory.map_url && memory.context_type !== 'online' ? (
                <Pressable onPress={() => Linking.openURL(memory.map_url!)}>
                  <Text style={styles.mapLink}>View where this was taken ↗</Text>
                </Pressable>
              ) : null}
              <View style={styles.detailAction}>
                <PrimaryButton label="Edit memory" onPress={() => setEditing(true)} />
              </View>
            </View>
          ) : (
            <View style={styles.editorBody}>
              <View style={styles.editorSection}>
                <Text style={styles.editorLabel}>Your note</Text>
                <TextInput
                  value={note}
                  onChangeText={setNote}
                  placeholder="What do you want to remember?"
                  placeholderTextColor="#8B8B85"
                  multiline
                  style={styles.editNoteInput}
                />
              </View>

              <View style={styles.editorSection}>
                <Text style={styles.editorLabel}>Tags</Text>
                <Text style={styles.editorHelp}>Tap one to remove it.</Text>
                {tags.length ? (
                  <View style={styles.editorChipWrap}>
                    {tags.map((tag) => (
                      <Pressable key={tag} onPress={() => setTags((current) => current.filter((item) => item !== tag))}>
                        <Text style={styles.editTagChip}>#{tag}  ×</Text>
                      </Pressable>
                    ))}
                  </View>
                ) : null}
                <View style={styles.editorAddRow}>
                  <TextInput
                    value={tagDraft}
                    onChangeText={setTagDraft}
                    onSubmitEditing={() => addTag()}
                    placeholder="Add a tag"
                    placeholderTextColor="#8B8B85"
                    autoCapitalize="none"
                    returnKeyType="done"
                    style={styles.editorSmallInput}
                  />
                  <Pressable onPress={() => addTag()} disabled={!normalizeTag(tagDraft)} style={styles.editorAddButton}>
                    <Ionicons name="add" size={20} color="#FFFFFF" />
                  </Pressable>
                </View>
                {usedTagSuggestions.length ? (
                  <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.editorSuggestionRow}>
                    {usedTagSuggestions.map((tag) => (
                      <Pressable key={tag} onPress={() => addTag(tag)} style={styles.editorSuggestionChip}>
                        <Text style={styles.editorSuggestionText}>#{tag}</Text>
                      </Pressable>
                    ))}
                  </ScrollView>
                ) : null}
              </View>

              <View style={styles.editorSection}>
                <Text style={styles.editorLabel}>Mood</Text>
                <Text style={styles.editorHelp}>Choose as many as fit.</Text>
                <View style={styles.editorChipWrap}>
                  {moodChoices.map((mood) => {
                    const selected = moods.includes(mood);
                    return (
                      <Pressable
                        key={mood}
                        onPress={() => setMoods((current) => selected ? current.filter((item) => item !== mood) : [...current, mood])}
                        style={[styles.choiceChip, selected && styles.choiceChipSelected]}
                      >
                        <Text style={[styles.choiceChipText, selected && styles.choiceChipTextSelected]}>{mood}</Text>
                      </Pressable>
                    );
                  })}
                </View>
                <View style={styles.editorAddRow}>
                  <TextInput
                    value={moodDraft}
                    onChangeText={setMoodDraft}
                    onSubmitEditing={() => addMood()}
                    placeholder="Add your own mood"
                    placeholderTextColor="#8B8B85"
                    returnKeyType="done"
                    style={styles.editorSmallInput}
                  />
                  <Pressable onPress={() => addMood()} disabled={!normalizeMood(moodDraft)} style={styles.editorAddButton}>
                    <Ionicons name="add" size={20} color="#FFFFFF" />
                  </Pressable>
                </View>
              </View>

              <View style={styles.editorSection}>
                <Text style={styles.editorLabel}>Where does this belong?</Text>
                <Text style={styles.editorHelp}>Online memories will stay out of nearby-place results.</Text>
                <View style={styles.editorChipWrap}>
                  {CONTEXT_OPTIONS.map((option) => {
                    const selected = contextType === option.value;
                    return (
                      <Pressable
                        key={option.value}
                        onPress={() => setContextType(option.value)}
                        style={[styles.choiceChip, selected && styles.contextChoiceSelected]}
                      >
                        <Ionicons name={option.icon} size={15} color={selected ? '#765D45' : '#666660'} />
                        <Text style={[styles.choiceChipText, selected && styles.contextChoiceText]}>{option.label}</Text>
                      </Pressable>
                    );
                  })}
                </View>
              </View>

              <View style={styles.editorSection}>
                <Text style={styles.editorLabel}>Shopping stage</Text>
                <Text style={styles.editorHelp}>Optional — change it as the story moves along.</Text>
                <View style={styles.editorChipWrap}>
                  {SHOPPING_OPTIONS.map((option) => {
                    const selected = shoppingStatus === option.value;
                    return (
                      <Pressable
                        key={option.value}
                        onPress={() => setShoppingStatus(option.value)}
                        style={[styles.choiceChip, selected && styles.statusChoiceSelected]}
                      >
                        <Text style={[styles.choiceChipText, selected && styles.statusChoiceText]}>{option.label}</Text>
                      </Pressable>
                    );
                  })}
                  {shoppingStatus ? (
                    <Pressable onPress={() => setShoppingStatus(null)} style={styles.clearChoiceChip}>
                      <Text style={styles.clearChoiceText}>Clear</Text>
                    </Pressable>
                  ) : null}
                </View>
              </View>

              {saveError ? <Text style={styles.editorError}>{saveError}</Text> : null}
              <PrimaryButton label={saving ? 'Saving…' : 'Save changes'} onPress={saveMemory} disabled={saving} />
              <Pressable style={styles.cancelEditButton} onPress={() => setEditing(false)} disabled={saving}>
                <Text style={styles.cancelEditText}>Cancel</Text>
              </Pressable>
            </View>
          )}
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

function SettingsScreen(props: ConnectionProps & { onBack: () => void }) {
  const { onBack, ...connectionProps } = props;
  return (
    <View style={styles.fill}>
      <PageHeader title="Settings" subtitle="Private connection" onBack={onBack} />
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.settingsContent} keyboardShouldPersistTaps="handled">
          <ConnectionForm {...connectionProps} />
          <NoteCard
            title="Automatic photo history"
            text="Gallery photos keep their original date and location when available. New camera photos use the current time and location. Bepo never substitutes your current location for an old photo."
          />
          <NoteCard
            title="Private by default"
            text="Your photos and memories live on your private Railway service. Your API key stays protected on this device."
          />
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

function PageHeader({ title, subtitle, onBack }: { title: string; subtitle: string; onBack: () => void }) {
  return (
    <View style={styles.pageHeader}>
      <Pressable accessibilityLabel="Back to Bepo" style={styles.backButton} onPress={onBack}>
        <Text style={styles.backButtonText}>‹</Text>
      </Pressable>
      <View style={styles.pageHeaderCopy}>
        <Text style={styles.pageHeaderTitle}>{title}</Text>
        <Text style={styles.pageHeaderSubtitle}>{subtitle}</Text>
      </View>
      <View style={styles.pageHeaderSpacer} />
    </View>
  );
}

function BrandMark({ size }: { size: number }) {
  return <Image source={require('../assets/bepo-bunny-icon.png')} style={{ width: size, height: size, borderRadius: Math.round(size * 0.3) }} />;
}

type ConnectionProps = {
  url: string;
  setUrl: (value: string) => void;
  apiKey: string;
  setApiKey: (value: string) => void;
  error: string;
  loading: boolean;
  onSave: () => void;
};

function ConnectionForm({ url, setUrl, apiKey, setApiKey, error, loading, onSave }: ConnectionProps) {
  return (
    <View style={styles.connectionCard}>
      <Field label="Bepo server" value={url} onChangeText={setUrl} autoCapitalize="none" keyboardType="url" />
      <Field label="Private API key" value={apiKey} onChangeText={setApiKey} autoCapitalize="none" secureTextEntry placeholder="Paste BEPO_API_KEY" />
      {error ? <Text style={styles.errorText}>{error}</Text> : null}
      <PrimaryButton label={loading ? 'Checking…' : 'Save connection'} onPress={onSave} disabled={loading} />
    </View>
  );
}

function NoteCard({ title, text }: { title: string; text: string }) {
  return (
    <View style={styles.noteCard}>
      <Text style={styles.noteCardTitle}>{title}</Text>
      <Text style={styles.noteCardText}>{text}</Text>
    </View>
  );
}

function Field({ label, ...inputProps }: { label: string } & React.ComponentProps<typeof TextInput>) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput {...inputProps} style={styles.input} placeholderTextColor="#8B8B85" />
    </View>
  );
}

function PrimaryButton({ label, onPress, disabled = false }: { label: string; onPress: () => void; disabled?: boolean }) {
  return (
    <Pressable style={[styles.primaryButton, disabled && styles.disabledButton]} onPress={onPress} disabled={disabled}>
      <Text style={styles.primaryButtonText}>{label}</Text>
    </Pressable>
  );
}

