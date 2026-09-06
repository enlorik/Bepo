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

type Screen = 'chat' | 'memories' | 'memory' | 'places' | 'settings';
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
  place_id?: number | null;
  lat: number | null;
  lon: number | null;
  location_source?: 'photo' | 'current' | 'manual' | null;
  image_url: string;
  map_url: string | null;
  distance_km?: number | null;
  score?: number;
};

type PlacePathItem = { id: number; name: string };

type PlaceChoice = {
  id: number;
  name: string;
  path: PlacePathItem[];
  path_label: string;
};

type Place = PlaceChoice & {
  parent_id: number | null;
  lat: number | null;
  lon: number | null;
  effective_lat: number | null;
  effective_lon: number | null;
  pin_inherited: boolean;
  path: PlacePathItem[];
  path_label: string;
  direct_memory_count: number;
  memory_count: number;
  child_count: number;
  distance_m?: number;
};

type PlaceDetail = Place & { children: Place[]; memories: Memory[] };

type MemoryFilters = {
  tags?: string[];
  moods?: string[];
  context_type?: MemoryContext;
  shopping_status?: ShoppingStatus;
  nearby?: boolean;
  place?: PlaceChoice;
};

type ChatResponse = {
  status: string;
  answer: string;
  memories: Memory[];
  filters?: MemoryFilters;
  place_options?: PlaceChoice[];
  suggested_place_name?: string;
};
type Requester = (path: string, options?: RequestInit) => Promise<any>;

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  photoUri?: string;
  tags?: string[];
  moods?: string[];
  placeLabel?: string;
  memories?: Memory[];
  filters?: MemoryFilters;
  placeOptions?: PlaceChoice[];
  originalQuery?: string;
  suggestedPlaceName?: string;
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

function isNearbyQuery(value: string) {
  return /\b(?:nearby|closest)\b|\bnear\s+me\b/i.test(value);
}

function formatDistance(distanceKm: number) {
  if (distanceKm < 1) return `${Math.max(1, Math.round(distanceKm * 1000))} m away`;
  return `${distanceKm < 10 ? distanceKm.toFixed(1) : Math.round(distanceKm)} km away`;
}

function placeForMemory(memory: Memory, places: Place[]) {
  return places.find((place) => place.id === memory.place_id) || null;
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

function normalizePlaceTerm(value: string) {
  return value
    .trim()
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .toLocaleLowerCase();
}

function activePlaceMention(value: string) {
  const match = value.match(/(?:^|\s)@([\p{L}\p{N}_-]*)$/u);
  if (!match) return null;
  return {
    query: normalizePlaceTerm(match[1]),
    start: value.lastIndexOf('@'),
  };
}

function detectImplicitPlace(value: string, places: Place[]): PlaceChoice | null {
  if (!value.trim() || activePlaceMention(value)) return null;
  const normalizedQuery = ` ${normalizePlaceTerm(value.replace(/[^\p{L}\p{N}_-]+/gu, ' '))} `;
  const groups = new Map<string, Place[]>();
  places.forEach((place) => {
    const normalizedName = normalizePlaceTerm(place.name);
    if (!normalizedName || !normalizedQuery.includes(` ${normalizedName} `)) return;
    groups.set(normalizedName, [...(groups.get(normalizedName) || []), place]);
  });
  const rankedGroups = [...groups.entries()].sort((left, right) => {
    const leftDepth = Math.max(...left[1].map((place) => place.path.length));
    const rightDepth = Math.max(...right[1].map((place) => place.path.length));
    return right[0].split(' ').length - left[0].split(' ').length
      || right[0].length - left[0].length
      || rightDepth - leftDepth;
  });
  if (!rankedGroups.length || rankedGroups[0][1].length !== 1) return null;
  return rankedGroups[0][1][0];
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
  const [selectedPlaceId, setSelectedPlaceId] = useState<number | null>(null);
  const [memoryReturnScreen, setMemoryReturnScreen] = useState<'memories' | 'places'>('memories');
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
  const [places, setPlaces] = useState<Place[]>([]);
  const [loadingPlaces, setLoadingPlaces] = useState(false);
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

  const loadPlaces = useCallback(
    async (quiet = false) => {
      if (!apiKey) return;
      if (!quiet) setLoadingPlaces(true);
      try {
        setPlaces((await request('/places')) as Place[]);
      } catch (error) {
        if (!quiet) Alert.alert('Could not load places', errorMessage(error));
      } finally {
        setLoadingPlaces(false);
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
    if (apiKey && !hydrating) {
      loadMemories(true);
      loadPlaces(true);
    }
  }, [apiKey, hydrating, loadMemories, loadPlaces]);

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

  function openMemory(memoryId: number, returnScreen: 'memories' | 'places' = 'memories') {
    setSelectedMemoryId(memoryId);
    setMemoryReturnScreen(returnScreen);
    setScreen('memory');
  }

  function updateMemory(updatedMemory: Memory) {
    setMemories((current) => current.map((memory) => (
      memory.id === updatedMemory.id ? updatedMemory : memory
    )));
    loadPlaces(true);
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
          places={places}
          onMemorySaved={() => loadMemories(true)}
          onOpenMemories={() => setScreen('memories')}
          onOpenPlaces={() => setScreen('places')}
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
          places={places}
          onBack={() => setScreen(memoryReturnScreen)}
          onSaved={updateMemory}
        />
      ) : null}
      {screen === 'places' ? (
        <PlacesScreen
          places={places}
          loading={loadingPlaces}
          selectedPlaceId={selectedPlaceId}
          setSelectedPlaceId={setSelectedPlaceId}
          request={request}
          apiUrl={apiUrl}
          apiKey={apiKey}
          onReload={() => loadPlaces(true)}
          onOpenMemory={(memoryId) => openMemory(memoryId, 'places')}
          onBack={() => setScreen('chat')}
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
  places,
  onMemorySaved,
  onOpenMemories,
  onOpenPlaces,
  onOpenSettings,
}: {
  request: Requester;
  apiUrl: string;
  apiKey: string;
  memoryCount: number;
  knownTags: string[];
  knownMoods: string[];
  places: Place[];
  onMemorySaved: () => Promise<void>;
  onOpenMemories: () => void;
  onOpenPlaces: () => void;
  onOpenSettings: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [pendingPhoto, setPendingPhoto] = useState<PendingPhoto | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [selectedMoods, setSelectedMoods] = useState<string[]>([]);
  const [selectedPlace, setSelectedPlace] = useState<PlaceChoice | null>(null);
  const [placeSelectionMode, setPlaceSelectionMode] = useState<'automatic' | 'manual' | null>(null);
  const [ignoredAutomaticPlaceId, setIgnoredAutomaticPlaceId] = useState<number | null>(null);
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
      if (placeSelectionMode !== 'manual') {
        const detected = detectImplicitPlace(value, places);
        if (detected && detected.id !== ignoredAutomaticPlaceId) {
          setSelectedPlace(detected);
          setPlaceSelectionMode('automatic');
        } else {
          setSelectedPlace(null);
          setPlaceSelectionMode(null);
          if (!detected) setIgnoredAutomaticPlaceId(null);
        }
      }
      return;
    }
    const consumed = consumeCompletedHashtags(value);
    setDraft(consumed.text);
    if (consumed.tags.length) addTags(consumed.tags);
  }

  function selectTag(tag: string) {
    const active = activeHashtag(draft);
    if (active && !pendingPhoto) {
      setDraft(`${draft.slice(0, active.start)}#${tag} `);
    } else {
      if (active) setDraft(draft.slice(0, active.start).trimEnd());
      addTags([tag]);
    }
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  function removeTag(tag: string) {
    setSelectedTags((current) => current.filter((item) => item !== tag));
  }

  function selectPlace(place: PlaceChoice) {
    const active = activePlaceMention(draft);
    if (active) setDraft(draft.slice(0, active.start).trimEnd());
    setSelectedPlace(place);
    setPlaceSelectionMode('manual');
    setIgnoredAutomaticPlaceId(null);
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  function removeSelectedPlace() {
    if (selectedPlace && placeSelectionMode === 'automatic') {
      setIgnoredAutomaticPlaceId(selectedPlace.id);
    }
    setSelectedPlace(null);
    setPlaceSelectionMode(null);
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

  async function sendMessage(textOverride?: string, placeOverride?: PlaceChoice) {
    const photo = pendingPhoto;
    const place = placeOverride ?? selectedPlace;
    const rawText = (textOverride ?? draft).trim();
    const taggedNote = photo ? finalizeTaggedNote(rawText, selectedTags) : { note: rawText, tags: [] };
    const text = taggedNote.note;
    const queryText = text || (place ? `Show me memories from ${place.name}` : '');
    const tags = taggedNote.tags;
    const moods = photo ? selectedMoods : [];
    if (sending || photo?.metadataLoading || (!queryText && !photo)) return;

    const userId = messageId('user');
    addMessage({
      id: userId,
      role: 'user',
      text: text || (photo ? 'Remember this.' : queryText),
      photoUri: photo?.asset.uri,
      tags: tags.length ? tags : undefined,
      moods: moods.length ? moods : undefined,
      placeLabel: place?.path_label,
      meta: photo ? 'Saving memory…' : undefined,
    });
    if (textOverride === undefined) setDraft('');
    setPendingPhoto(null);
    setSelectedTags([]);
    setSelectedMoods([]);
    setSelectedPlace(null);
    setPlaceSelectionMode(null);
    const detectPlaces = ignoredAutomaticPlaceId === null;
    setIgnoredAutomaticPlaceId(null);
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
        if (place) form.append('place_id', String(place.id));
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
        let coordinates: Coordinates | null = null;
        if (isNearbyQuery(text)) {
          updateMessage(userId, { meta: 'Finding what is closest…' });
          coordinates = await getAutomaticLocation();
          updateMessage(userId, { meta: coordinates ? 'Used your current location' : 'Location was not available' });
        }
        const response = (await request('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: queryText,
            top_k: 5,
            lat: coordinates?.lat,
            lon: coordinates?.lon,
            place_id: place?.id,
            detect_places: detectPlaces,
          }),
        })) as ChatResponse;
        addMessage({
          id: messageId('bepo'),
          role: 'assistant',
          text: response.answer,
          memories: response.memories || [],
          filters: response.filters,
          placeOptions: response.place_options,
          originalQuery: queryText,
          suggestedPlaceName: response.suggested_place_name,
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

  const canSend = Boolean(draft.trim() || pendingPhoto || selectedPlace) && !sending && !pendingPhoto?.metadataLoading;
  const activeTag = activeHashtag(draft);
  const activePlace = activePlaceMention(draft);
  const matchingTags = activeTag
    ? knownTags
      .filter((tag) => (!pendingPhoto || !selectedTags.includes(tag)) && (!activeTag.query || tag.startsWith(activeTag.query)))
      .slice(0, 6)
    : [];
  const canCreateTag = Boolean(
    pendingPhoto
    &&
    activeTag?.query
    && !selectedTags.includes(activeTag.query)
    && !knownTags.includes(activeTag.query),
  );
  const matchingPlaces = activePlace
    ? places
      .filter((place) => {
        if (!activePlace.query) return true;
        return normalizePlaceTerm(place.name).includes(activePlace.query)
          || normalizePlaceTerm(place.path_label.replace(/›/g, ' ')).includes(activePlace.query);
      })
      .sort((left, right) => {
        const leftStarts = normalizePlaceTerm(left.name).startsWith(activePlace.query) ? 0 : 1;
        const rightStarts = normalizePlaceTerm(right.name).startsWith(activePlace.query) ? 0 : 1;
        return leftStarts - rightStarts || left.path_label.localeCompare(right.path_label);
      })
      .slice(0, 8)
    : [];
  const moodOptions = [...new Set([...knownMoods, ...STARTER_MOODS, ...selectedMoods])];

  return (
    <View style={styles.fill}>
      <ChatControls
        memoryCount={memoryCount}
        onOpenMemories={onOpenMemories}
        onOpenPlaces={onOpenPlaces}
        onOpenSettings={onOpenSettings}
      />
      <KeyboardAvoidingView
        style={styles.fill}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 38 : 0}
      >
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <ChatBubble
              message={item}
              apiUrl={apiUrl}
              apiKey={apiKey}
              onChoosePlace={(place) => sendMessage(item.originalQuery, place)}
              onOpenPlaces={onOpenPlaces}
            />
          )}
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
          {selectedPlace ? (
            <View style={styles.selectedPlaceRow}>
              <Pressable
                accessibilityLabel={`Remove place ${selectedPlace.path_label}`}
                style={styles.selectedPlaceChip}
                onPress={removeSelectedPlace}
              >
                <Ionicons name="location-outline" size={14} color="#4F6270" />
                <Text style={styles.selectedPlaceText}>{selectedPlace.path_label}</Text>
                <Ionicons name="close" size={13} color="#4F6270" />
              </Pressable>
            </View>
          ) : null}
          {activePlace ? (
            <View style={styles.placeMentionSuggestions}>
              {matchingPlaces.length ? (
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  keyboardShouldPersistTaps="always"
                  contentContainerStyle={styles.tagSuggestionRow}
                >
                  {matchingPlaces.map((place) => (
                    <Pressable
                      key={place.id}
                      accessibilityLabel={`Use place ${place.path_label}`}
                      style={styles.placeMentionChip}
                      onPress={() => selectPlace(place)}
                    >
                      <Ionicons name="location-outline" size={14} color="#4F6270" />
                      <Text style={styles.placeMentionText}>{place.path_label}</Text>
                    </Pressable>
                  ))}
                </ScrollView>
              ) : (
                <Text style={styles.tagSuggestionHint}>
                  {activePlace.query ? 'No saved place matches. Bepo can help you create it after searching.' : 'Create a place from the branching icon first.'}
                </Text>
              )}
            </View>
          ) : null}
          {activeTag ? (
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
                <Text style={styles.tagSuggestionHint}>{pendingPhoto ? 'Keep typing to make a new tag' : 'No saved tag matches yet'}</Text>
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
              placeholder={pendingPhoto ? 'Add a note… #tag or @place' : 'Message Bepo… #tag or @place'}
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

function ChatControls({ memoryCount, onOpenMemories, onOpenPlaces, onOpenSettings }: {
  memoryCount: number;
  onOpenMemories: () => void;
  onOpenPlaces: () => void;
  onOpenSettings: () => void;
}) {
  return (
    <View style={styles.chatControls}>
      <View style={styles.headerActions}>
        <Pressable accessibilityLabel="Open memories" style={styles.headerButton} onPress={onOpenMemories}>
          <Ionicons name="albums-outline" size={22} color="#393936" />
          {memoryCount ? <View style={styles.memoryBadge}><Text style={styles.memoryBadgeText}>{memoryCount > 99 ? '99+' : memoryCount}</Text></View> : null}
        </Pressable>
        <Pressable accessibilityLabel="Open places" style={styles.headerButton} onPress={onOpenPlaces}>
          <Ionicons name="git-branch-outline" size={21} color="#393936" />
        </Pressable>
        <Pressable accessibilityLabel="Open settings" style={styles.headerButton} onPress={onOpenSettings}>
          <Ionicons name="ellipsis-horizontal" size={23} color="#393936" />
        </Pressable>
      </View>
    </View>
  );
}

function EmptyConversation({ onPrompt }: { onPrompt: (prompt: string) => void }) {
  const prompts = ['#cafe calm nearby', '#shopping want', 'What do you remember most recently?'];
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

function ChatBubble({ message, apiUrl, apiKey, onChoosePlace, onOpenPlaces }: {
  message: ChatMessage;
  apiUrl: string;
  apiKey: string;
  onChoosePlace: (place: PlaceChoice) => void;
  onOpenPlaces: () => void;
}) {
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
          {message.placeLabel ? (
            <View style={styles.userPlaceRow}>
              <Text style={styles.userPlace}>⌖ {message.placeLabel}</Text>
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
        {message.filters && Object.keys(message.filters).length ? (
          <View style={styles.assistantFilterRow}>
            {message.filters.tags?.map((tag) => <Text key={`tag-${tag}`} style={styles.assistantTagFilter}>#{tag}</Text>)}
            {message.filters.moods?.map((mood) => <Text key={`mood-${mood}`} style={styles.assistantMoodFilter}>{mood}</Text>)}
            {message.filters.context_type ? <Text style={styles.assistantNeutralFilter}>{contextLabel(message.filters.context_type)}</Text> : null}
            {message.filters.shopping_status ? <Text style={styles.assistantStatusFilter}>{shoppingLabel(message.filters.shopping_status)}</Text> : null}
            {message.filters.nearby ? <Text style={styles.assistantNearbyFilter}>⌖ nearby</Text> : null}
            {message.filters.place ? <Text style={styles.assistantPlaceFilter}>⌖ {message.filters.place.path_label}</Text> : null}
          </View>
        ) : null}
        {message.placeOptions?.length ? (
          <View style={styles.placeOptionList}>
            {message.placeOptions.map((place) => (
              <Pressable
                key={place.id}
                accessibilityLabel={`Choose ${place.path_label}`}
                style={styles.placeOptionButton}
                onPress={() => onChoosePlace(place)}
              >
                <Ionicons name="location-outline" size={16} color="#4F6270" />
                <Text style={styles.placeOptionText}>{place.path_label}</Text>
                <Ionicons name="chevron-forward" size={15} color="#7B8982" />
              </Pressable>
            ))}
          </View>
        ) : null}
        {message.suggestedPlaceName ? (
          <Pressable style={styles.createPlaceSuggestion} onPress={onOpenPlaces}>
            <Ionicons name="add" size={16} color="#4F6270" />
            <Text style={styles.createPlaceSuggestionText}>Open Places to create “{message.suggestedPlaceName}”</Text>
          </Pressable>
        ) : null}
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

function PlacesScreen({ places, loading, selectedPlaceId, setSelectedPlaceId, request, apiUrl, apiKey, onReload, onOpenMemory, onBack }: {
  places: Place[];
  loading: boolean;
  selectedPlaceId: number | null;
  setSelectedPlaceId: (placeId: number | null) => void;
  request: Requester;
  apiUrl: string;
  apiKey: string;
  onReload: () => Promise<void>;
  onOpenMemory: (memoryId: number) => void;
  onBack: () => void;
}) {
  const [detail, setDetail] = useState<PlaceDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState('');
  const [editing, setEditing] = useState<{ place: Place | null; parentId: number | null } | null>(null);

  const loadDetail = useCallback(async (placeId: number) => {
    setDetailLoading(true);
    setDetailError('');
    try {
      setDetail((await request(`/places/${placeId}`)) as PlaceDetail);
    } catch (error) {
      setDetailError(errorMessage(error));
    } finally {
      setDetailLoading(false);
    }
  }, [request]);

  useEffect(() => {
    if (selectedPlaceId !== null) loadDetail(selectedPlaceId);
    else setDetail(null);
  }, [loadDetail, selectedPlaceId]);

  async function handlePlaceSaved(place: Place) {
    await onReload();
    setEditing(null);
    setSelectedPlaceId(place.id);
    await loadDetail(place.id);
  }

  if (editing) {
    return (
      <PlaceEditorScreen
        places={places}
        place={editing.place}
        initialParentId={editing.parentId}
        request={request}
        onSaved={handlePlaceSaved}
        onCancel={() => setEditing(null)}
      />
    );
  }

  if (selectedPlaceId !== null) {
    const current = detail || places.find((place) => place.id === selectedPlaceId) || null;
    const parentId = current?.parent_id ?? null;
    return (
      <View style={styles.fill}>
        <PageHeader
          title={current?.name || 'Place'}
          subtitle={current?.path.slice(0, -1).map((item) => item.name).join(' › ') || 'Your place forest'}
          onBack={() => setSelectedPlaceId(parentId)}
        />
        <ScrollView contentContainerStyle={styles.placeDetailContent} showsVerticalScrollIndicator={false}>
          {detailLoading && !detail ? <ActivityIndicator color="#262624" style={styles.placeLoader} /> : null}
          {detailError ? <Text style={styles.editorError}>{detailError}</Text> : null}
          {current ? (
            <>
              <View style={styles.placeHeroCard}>
                <View style={styles.placeHeroIcon}><Ionicons name="location" size={23} color="#765D45" /></View>
                <View style={styles.placeHeroCopy}>
                  <Text style={styles.placeHeroTitle}>{current.path_label}</Text>
                  <Text style={styles.placeHeroMeta}>
                    {current.lat !== null
                      ? 'Has its own map pin'
                      : current.pin_inherited
                        ? 'Uses its parent’s map pin'
                        : 'A folder without a map pin'}
                  </Text>
                  <Text style={styles.placeHeroMeta}>{current.memory_count} memories in this branch</Text>
                </View>
              </View>

              <View style={styles.placeActionRow}>
                <Pressable
                  style={styles.placeActionButton}
                  onPress={() => setEditing({ place: null, parentId: current.id })}
                >
                  <Ionicons name="add" size={18} color="#FFFFFF" />
                  <Text style={styles.placeActionButtonText}>Add inside</Text>
                </Pressable>
                <Pressable
                  style={styles.placeSecondaryAction}
                  onPress={() => setEditing({ place: current, parentId: current.parent_id })}
                >
                  <Ionicons name="pencil-outline" size={17} color="#555550" />
                  <Text style={styles.placeSecondaryActionText}>Edit</Text>
                </Pressable>
              </View>

              <Text style={styles.placeSectionTitle}>Places inside</Text>
              {detail?.children.length ? (
                <View style={styles.placeBranchList}>
                  {detail.children.map((child) => (
                    <PlaceBranchRow key={child.id} place={child} onPress={() => setSelectedPlaceId(child.id)} />
                  ))}
                </View>
              ) : (
                <Text style={styles.placeEmptyText}>Nothing nested here yet. “Add inside” can make a room, floor, café area, or anything else.</Text>
              )}

              <Text style={styles.placeSectionTitle}>Memories in this branch</Text>
              {detail?.memories.length ? (
                <View style={styles.placeMemoryList}>
                  {detail.memories.map((memory) => (
                    <MemoryCard
                      key={memory.id}
                      memory={memory}
                      apiUrl={apiUrl}
                      apiKey={apiKey}
                      onPress={() => onOpenMemory(memory.id)}
                    />
                  ))}
                </View>
              ) : (
                <Text style={styles.placeEmptyText}>No memories here yet. Open a memory, tap Edit memory, then choose this place.</Text>
              )}
            </>
          ) : null}
        </ScrollView>
      </View>
    );
  }

  const roots = places.filter((place) => place.parent_id === null);
  return (
    <View style={styles.fill}>
      <PageHeader title="Places" subtitle="Your memory forest" onBack={onBack} />
      <ScrollView contentContainerStyle={styles.placesContent} showsVerticalScrollIndicator={false}>
        <View style={styles.forestIntro}>
          <View style={styles.forestIllustration}>
            <Ionicons name="git-branch" size={28} color="#4F6B5B" />
          </View>
          <View style={styles.forestIntroCopy}>
            <Text style={styles.forestIntroTitle}>Build places your way.</Text>
            <Text style={styles.forestIntroText}>A pin can hold smaller places forever: Home → Kitchen, or Square → Cat Café → Upstairs.</Text>
          </View>
        </View>
        <PrimaryButton label="Create a place" onPress={() => setEditing({ place: null, parentId: null })} />
        <Text style={styles.placeSectionTitle}>Your top-level places</Text>
        {loading && !places.length ? <ActivityIndicator color="#262624" style={styles.placeLoader} /> : null}
        {roots.length ? (
          <View style={styles.placeBranchList}>
            {roots.map((place) => (
              <PlaceBranchRow key={place.id} place={place} onPress={() => setSelectedPlaceId(place.id)} />
            ))}
          </View>
        ) : !loading ? (
          <View style={styles.emptyForest}>
            <BrandMark size={62} />
            <Text style={styles.emptyForestTitle}>Your forest starts with one place.</Text>
            <Text style={styles.placeEmptyText}>Try Home, a neighborhood, your favorite café, or an online world.</Text>
          </View>
        ) : null}
      </ScrollView>
    </View>
  );
}

function PlaceBranchRow({ place, onPress }: { place: Place; onPress: () => void }) {
  return (
    <Pressable style={({ pressed }) => [styles.placeBranchRow, pressed && styles.memoryCardPressed]} onPress={onPress}>
      <View style={styles.placeBranchIcon}>
        <Ionicons name={place.child_count ? 'git-branch-outline' : 'location-outline'} size={19} color="#765D45" />
      </View>
      <View style={styles.placeBranchCopy}>
        <Text style={styles.placeBranchName}>{place.name}</Text>
        <Text style={styles.placeBranchMeta}>
          {place.child_count ? `${place.child_count} inside · ` : ''}{place.memory_count} memories
        </Text>
      </View>
      <Ionicons name="chevron-forward" size={18} color="#9A9A94" />
    </Pressable>
  );
}

function PlaceEditorScreen({ places, place, initialParentId, request, onSaved, onCancel }: {
  places: Place[];
  place: Place | null;
  initialParentId: number | null;
  request: Requester;
  onSaved: (place: Place) => Promise<void>;
  onCancel: () => void;
}) {
  const [name, setName] = useState(place?.name || '');
  const [parentId, setParentId] = useState<number | null>(place?.parent_id ?? initialParentId);
  const [coordinates, setCoordinates] = useState<Coordinates | null>(
    place?.lat !== null && place?.lat !== undefined && place.lon !== null
      ? { lat: place.lat, lon: place.lon }
      : null,
  );
  const [locating, setLocating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const parentChoices = places.filter((candidate) => (
    !place || (candidate.id !== place.id && !candidate.path.some((item) => item.id === place.id))
  ));
  const selectedParent = places.find((candidate) => candidate.id === parentId) || null;

  async function useCurrentLocation() {
    setLocating(true);
    setError('');
    try {
      let permission = await Location.getForegroundPermissionsAsync();
      if (!permission.granted && permission.canAskAgain) {
        permission = await Location.requestForegroundPermissionsAsync();
      }
      if (!permission.granted) throw new Error('Allow location access to pin this place. You can also leave it as a folder.');
      const current = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
      setCoordinates({ lat: current.coords.latitude, lon: current.coords.longitude });
    } catch (locationError) {
      setError(errorMessage(locationError));
    } finally {
      setLocating(false);
    }
  }

  async function savePlace() {
    if (!name.trim()) {
      setError('Give this place a name first.');
      return;
    }
    setSaving(true);
    setError('');
    try {
      const saved = await request(place ? `/places/${place.id}` : '/places', {
        method: place ? 'PATCH' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          parent_id: parentId,
          lat: coordinates?.lat ?? null,
          lon: coordinates?.lon ?? null,
        }),
      }) as Place;
      await onSaved(saved);
    } catch (saveError) {
      setError(errorMessage(saveError));
    } finally {
      setSaving(false);
    }
  }

  return (
    <View style={styles.fill}>
      <PageHeader title={place ? 'Edit place' : 'New place'} subtitle="You stay in control" onBack={onCancel} />
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.placeEditorContent} keyboardShouldPersistTaps="handled">
          <View style={styles.editorSection}>
            <Text style={styles.editorLabel}>Name</Text>
            <TextInput
              value={name}
              onChangeText={setName}
              placeholder="Home, Cat Café, Upstairs…"
              placeholderTextColor="#8B8B85"
              autoFocus={!place}
              style={styles.placeNameInput}
            />
          </View>

          <View style={styles.editorSection}>
            <Text style={styles.editorLabel}>Put it inside</Text>
            <Text style={styles.editorHelp}>Choose a parent, or leave it at the top of your forest.</Text>
            <Pressable
              onPress={() => setParentId(null)}
              style={[styles.placeChoiceRow, parentId === null && styles.placeChoiceRowSelected]}
            >
              <Ionicons name={parentId === null ? 'checkmark-circle' : 'ellipse-outline'} size={18} color={parentId === null ? '#4F6B5B' : '#A0A09A'} />
              <Text style={[styles.placeChoiceRowText, parentId === null && styles.placeChoiceRowTextSelected]}>Top level</Text>
            </Pressable>
            {parentChoices.map((candidate) => {
              const selected = parentId === candidate.id;
              return (
                <Pressable
                  key={candidate.id}
                  onPress={() => setParentId(candidate.id)}
                  style={[styles.placeChoiceRow, selected && styles.placeChoiceRowSelected]}
                >
                  <Ionicons name={selected ? 'checkmark-circle' : 'ellipse-outline'} size={18} color={selected ? '#4F6B5B' : '#A0A09A'} />
                  <Text style={[styles.placeChoiceRowText, selected && styles.placeChoiceRowTextSelected]}>{candidate.path_label}</Text>
                </Pressable>
              );
            })}
          </View>

          <View style={styles.editorSection}>
            <Text style={styles.editorLabel}>Map pin</Text>
            <Text style={styles.editorHelp}>
              {coordinates
                ? 'This place has its own GPS pin.'
                : selectedParent?.effective_lat !== null && selectedParent?.effective_lat !== undefined
                  ? `It will inherit ${selectedParent.name}’s pin unless you add its own.`
                  : 'Optional. Folders and indoor areas work without their own GPS.'}
            </Text>
            {coordinates ? (
              <View style={styles.savedPinRow}>
                <Ionicons name="location" size={18} color="#4F6B5B" />
                <Text style={styles.savedPinText}>Pin ready</Text>
                <Pressable onPress={() => setCoordinates(null)}><Text style={styles.removePinText}>Remove</Text></Pressable>
              </View>
            ) : (
              <Pressable style={styles.useLocationButton} onPress={useCurrentLocation} disabled={locating}>
                {locating ? <ActivityIndicator size="small" color="#4F6B5B" /> : <Ionicons name="locate-outline" size={18} color="#4F6B5B" />}
                <Text style={styles.useLocationText}>{locating ? 'Finding you…' : 'Use my current location'}</Text>
              </Pressable>
            )}
          </View>

          {error ? <Text style={styles.editorError}>{error}</Text> : null}
          <PrimaryButton label={saving ? 'Saving…' : place ? 'Save place' : 'Create place'} onPress={savePlace} disabled={saving} />
          <Pressable style={styles.cancelEditButton} onPress={onCancel} disabled={saving}>
            <Text style={styles.cancelEditText}>Cancel</Text>
          </Pressable>
        </ScrollView>
      </KeyboardAvoidingView>
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
          {memory.distance_km !== null && memory.distance_km !== undefined ? (
            <Text style={styles.memoryDistance}>⌖ {formatDistance(memory.distance_km)}</Text>
          ) : null}
        </View>
        {memory.place_hint ? <Text style={styles.memoryPlace}>⌖ {memory.place_hint}</Text> : null}
        <Text style={styles.memoryHistory}>{memoryHistoryText(memory)}</Text>
        {memory.map_url && memory.context_type !== 'online' ? (
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
  places,
  onBack,
  onSaved,
}: {
  memory: Memory;
  apiUrl: string;
  apiKey: string;
  request: Requester;
  knownTags: string[];
  knownMoods: string[];
  places: Place[];
  onBack: () => void;
  onSaved: (memory: Memory) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [note, setNote] = useState(memory.user_note || memory.note || memory.bepo_summary || '');
  const [tags, setTags] = useState(splitStoredTags(memory.tags));
  const [moods, setMoods] = useState(splitStoredMoods(memory.mood));
  const [contextType, setContextType] = useState<MemoryContext>(memory.context_type || 'unknown');
  const [shoppingStatus, setShoppingStatus] = useState<ShoppingStatus | null>(memory.shopping_status || null);
  const [placeId, setPlaceId] = useState<number | null>(memory.place_id || null);
  const [placeSuggestions, setPlaceSuggestions] = useState<Place[]>([]);
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
    setPlaceId(memory.place_id || null);
    setTagDraft('');
    setMoodDraft('');
    setSaveError('');
  }, [memory.id]);

  useEffect(() => {
    let active = true;
    if (memory.lat === null || memory.lon === null) {
      setPlaceSuggestions([]);
      return () => { active = false; };
    }
    request(`/places/suggestions?lat=${encodeURIComponent(memory.lat)}&lon=${encodeURIComponent(memory.lon)}`)
      .then((suggestions: Place[]) => {
        if (active) setPlaceSuggestions(suggestions);
      })
      .catch(() => {
        if (active) setPlaceSuggestions([]);
      });
    return () => { active = false; };
  }, [memory.id, memory.lat, memory.lon, request]);

  const description = memory.user_note || memory.bepo_summary || memory.note || 'A saved moment';
  const assignedPlace = placeForMemory(memory, places);
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
          place_id: placeId,
        }),
      }) as Memory;
      onSaved(updated);
      setNote(updated.user_note || updated.note || updated.bepo_summary || '');
      setTags(splitStoredTags(updated.tags));
      setMoods(splitStoredMoods(updated.mood));
      setContextType(updated.context_type || 'unknown');
      setShoppingStatus(updated.shopping_status || null);
      setPlaceId(updated.place_id || null);
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
                {assignedPlace ? <Text style={styles.placeBadge}>⌖ {assignedPlace.path_label}</Text> : null}
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
                <Text style={styles.editorLabel}>Place</Text>
                <Text style={styles.editorHelp}>You decide where this belongs. Bepo only suggests existing nearby pins.</Text>
                {placeSuggestions.length ? (
                  <>
                    <Text style={styles.placeChoiceHeading}>Nearby suggestions</Text>
                    <View style={styles.editorChipWrap}>
                      {placeSuggestions.map((place) => {
                        const selected = placeId === place.id;
                        return (
                          <Pressable
                            key={`suggested-${place.id}`}
                            onPress={() => setPlaceId(place.id)}
                            style={[styles.placeChoice, selected && styles.placeChoiceSelected]}
                          >
                            <Ionicons name="location-outline" size={15} color={selected ? '#765D45' : '#65655F'} />
                            <View style={styles.placeChoiceCopy}>
                              <Text style={[styles.placeChoiceName, selected && styles.placeChoiceNameSelected]}>{place.path_label}</Text>
                              {place.distance_m !== undefined ? <Text style={styles.placeChoiceMeta}>{place.distance_m} m away</Text> : null}
                            </View>
                          </Pressable>
                        );
                      })}
                    </View>
                  </>
                ) : null}
                {places.length ? (
                  <>
                    <Text style={styles.placeChoiceHeading}>All places</Text>
                    <View style={styles.placeChoiceList}>
                      {places.map((place) => {
                        const selected = placeId === place.id;
                        return (
                          <Pressable
                            key={place.id}
                            onPress={() => setPlaceId(place.id)}
                            style={[styles.placeChoiceRow, selected && styles.placeChoiceRowSelected]}
                          >
                            <Ionicons name={selected ? 'checkmark-circle' : 'ellipse-outline'} size={18} color={selected ? '#4F6B5B' : '#A0A09A'} />
                            <Text style={[styles.placeChoiceRowText, selected && styles.placeChoiceRowTextSelected]}>{place.path_label}</Text>
                          </Pressable>
                        );
                      })}
                    </View>
                    {placeId ? (
                      <Pressable onPress={() => setPlaceId(null)} style={styles.clearPlaceButton}>
                        <Text style={styles.clearChoiceText}>Remove from this place</Text>
                      </Pressable>
                    ) : null}
                  </>
                ) : (
                  <Text style={styles.emptyPlaceHelp}>Create your first place from the branching icon on Bepo’s home screen.</Text>
                )}
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

