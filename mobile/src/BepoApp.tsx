import { StatusBar } from 'expo-status-bar';
import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import * as SecureStore from 'expo-secure-store';
import React, { useCallback, useEffect, useState } from 'react';
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

type Screen = 'home' | 'add' | 'search' | 'ask' | 'settings';

type Memory = {
  id: number;
  timestamp: string;
  note: string | null;
  user_note?: string | null;
  bepo_summary?: string | null;
  tags?: string | null;
  mood?: string | null;
  place_hint?: string | null;
  lat: number | null;
  lon: number | null;
  image_url: string;
  map_url: string | null;
  score?: number;
};

type ChatResponse = { status: string; answer: string; memories: Memory[] };
type SearchResponse = { status: string; matches: Memory[] };
type Requester = (path: string, options?: RequestInit) => Promise<any>;

const tabs: Array<{ id: Screen; label: string; glyph: string }> = [
  { id: 'home', label: 'Memories', glyph: '◫' },
  { id: 'add', label: 'Add', glyph: '+' },
  { id: 'search', label: 'Search', glyph: '⌕' },
  { id: 'ask', label: 'Ask', glyph: '✦' },
  { id: 'settings', label: 'Settings', glyph: '⚙' },
];

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

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
}

export default function BepoApp() {
  const [screen, setScreen] = useState<Screen>('home');
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
      setScreen('home');
    } catch (error) {
      setConnectionError(errorMessage(error));
    } finally {
      setConnecting(false);
    }
  }

  if (hydrating) {
    return (
      <View style={styles.centeredPage}>
        <StatusBar style="dark" />
        <BrandMark />
        <ActivityIndicator color="#715840" style={{ marginTop: 20 }} />
      </View>
    );
  }

  if (!apiKey) {
    return (
      <KeyboardAvoidingView style={styles.setupPage} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <StatusBar style="dark" />
        <ScrollView contentContainerStyle={styles.setupContent} keyboardShouldPersistTaps="handled">
          <BrandMark />
          <Text style={styles.setupEyebrow}>YOUR PRIVATE MEMORY SPACE</Text>
          <Text style={styles.setupTitle}>Meet Bepo.</Text>
          <Text style={styles.setupCopy}>
            Your Bepo server is already online. Connect this phone once and your key stays protected on this device.
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
      <View style={styles.page}>
        {screen === 'home' && (
          <HomeScreen
            memories={memories}
            apiUrl={apiUrl}
            apiKey={apiKey}
            loading={loadingMemories}
            refreshing={refreshing}
            onRefresh={() => { setRefreshing(true); loadMemories(true); }}
            onAdd={() => setScreen('add')}
          />
        )}
        {screen === 'add' && (
          <AddScreen request={request} onSaved={async () => { await loadMemories(true); setScreen('home'); }} />
        )}
        {screen === 'search' && <SearchScreen request={request} apiUrl={apiUrl} apiKey={apiKey} />}
        {screen === 'ask' && <AskScreen request={request} apiUrl={apiUrl} apiKey={apiKey} />}
        {screen === 'settings' && (
          <SettingsScreen
            url={draftUrl}
            setUrl={setDraftUrl}
            apiKey={draftKey}
            setApiKey={setDraftKey}
            error={connectionError}
            loading={connecting}
            onSave={saveConnection}
          />
        )}
      </View>
      <View style={styles.tabBar}>
        {tabs.map((tab) => (
          <Pressable key={tab.id} style={styles.tab} onPress={() => setScreen(tab.id)}>
            <Text style={[styles.tabGlyph, screen === tab.id && styles.tabGlyphActive]}>{tab.glyph}</Text>
            <Text style={[styles.tabLabel, screen === tab.id && styles.tabLabelActive]}>{tab.label}</Text>
          </Pressable>
        ))}
      </View>
    </View>
  );
}

function BrandMark() {
  return <Image source={require('../assets/bepo-bunny-icon.png')} style={styles.brandMarkImage} />;
}

function Header({ eyebrow, title, subtitle }: { eyebrow: string; title: string; subtitle?: string }) {
  return (
    <View style={styles.header}>
      <Text style={styles.eyebrow}>{eyebrow}</Text>
      <Text style={styles.title}>{title}</Text>
      {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
    </View>
  );
}

function HomeScreen({ memories, apiUrl, apiKey, loading, refreshing, onRefresh, onAdd }: {
  memories: Memory[]; apiUrl: string; apiKey: string; loading: boolean; refreshing: boolean; onRefresh: () => void; onAdd: () => void;
}) {
  return (
    <View style={styles.fill}>
      <FlatList
        data={memories}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={memories.length ? styles.listContent : styles.emptyListContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#715840" />}
        ListHeaderComponent={<Header eyebrow="BEPO" title="Your memories" subtitle={memories.length ? `${memories.length} saved moment${memories.length === 1 ? '' : 's'}` : undefined} />}
        ListEmptyComponent={loading ? <ActivityIndicator color="#715840" /> : (
          <View style={styles.emptyCard}>
            <Text style={styles.emptyGlyph}>✦</Text>
            <Text style={styles.emptyTitle}>Start with one moment.</Text>
            <Text style={styles.emptyCopy}>Choose a photo, add what you remember, and Bepo will keep it close.</Text>
            <PrimaryButton label="Add your first memory" onPress={onAdd} />
          </View>
        )}
        renderItem={({ item }) => <MemoryCard memory={item} apiUrl={apiUrl} apiKey={apiKey} />}
      />
    </View>
  );
}

function MemoryCard({ memory, apiUrl, apiKey }: { memory: Memory; apiUrl: string; apiKey: string }) {
  const description = memory.bepo_summary || memory.user_note || memory.note || 'A saved moment';
  return (
    <View style={styles.memoryCard}>
      <Image source={{ uri: absoluteUrl(apiUrl, memory.image_url), headers: { 'X-API-Key': apiKey } }} style={styles.memoryImage} resizeMode="cover" />
      <View style={styles.memoryBody}>
        <View style={styles.memoryMetaRow}>
          <Text style={styles.memoryDate}>{formatDate(memory.timestamp)}</Text>
          {memory.mood ? <Text style={styles.pill}>{memory.mood}</Text> : null}
        </View>
        <Text style={styles.memoryDescription}>{description}</Text>
        {memory.tags ? <Text style={styles.memoryTags}>{memory.tags}</Text> : null}
        {memory.place_hint ? <Text style={styles.memoryPlace}>⌖ {memory.place_hint}</Text> : null}
        {memory.map_url ? <Pressable onPress={() => Linking.openURL(memory.map_url!)}><Text style={styles.mapLink}>Open location ↗</Text></Pressable> : null}
        {typeof memory.score === 'number' ? <Text style={styles.score}>Match {Math.round(memory.score * 100)}%</Text> : null}
      </View>
    </View>
  );
}

function AddScreen({ request, onSaved }: { request: Requester; onSaved: () => void }) {
  const [photo, setPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [note, setNote] = useState('');
  const [tags, setTags] = useState('');
  const [mood, setMood] = useState('');
  const [placeHint, setPlaceHint] = useState('');
  const [coords, setCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [locating, setLocating] = useState(false);
  const [saving, setSaving] = useState(false);

  async function choosePhoto(source: 'camera' | 'library') {
    try {
      const permission = source === 'camera'
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Permission needed', `Allow ${source === 'camera' ? 'camera' : 'photo'} access to save this memory.`);
        return;
      }
      const result = source === 'camera'
        ? await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.85 })
        : await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85 });
      if (!result.canceled && result.assets[0]) setPhoto(result.assets[0]);
    } catch (error) {
      Alert.alert('Could not open photos', errorMessage(error));
    }
  }

  async function useLocation() {
    setLocating(true);
    try {
      const permission = await Location.requestForegroundPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Location permission needed', 'Allow location access to attach where this memory happened.');
        return;
      }
      const current = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
      setCoords({ lat: current.coords.latitude, lon: current.coords.longitude });
    } catch (error) {
      Alert.alert('Could not get location', errorMessage(error));
    } finally {
      setLocating(false);
    }
  }

  async function save() {
    if (!photo) {
      Alert.alert('Choose a photo', 'Every Bepo memory starts with a photo.');
      return;
    }
    setSaving(true);
    try {
      const form = new FormData();
      const photoFile = new File(photo.uri);
      form.append('photo', photoFile, photo.fileName || photoFile.name || `bepo-${Date.now()}.jpg`);
      if (note.trim()) form.append('note', note.trim());
      if (tags.trim()) form.append('tags', tags.trim());
      if (mood.trim()) form.append('mood', mood.trim());
      if (placeHint.trim()) form.append('place_hint', placeHint.trim());
      if (coords) {
        form.append('lat', String(coords.lat));
        form.append('lon', String(coords.lon));
      }
      await request('/memory', { method: 'POST', body: form });
      Alert.alert('Memory saved', 'Bepo will keep this one close.', [{ text: 'Done', onPress: onSaved }]);
    } catch (error) {
      Alert.alert('Could not save memory', errorMessage(error));
    } finally {
      setSaving(false);
    }
  }

  return (
    <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        <Header eyebrow="CAPTURE" title="Save a moment" subtitle="A photo is enough. Add details if they matter." />
        {photo ? (
          <Pressable onPress={() => choosePhoto('library')}>
            <Image source={{ uri: photo.uri }} style={styles.photoPreview} />
            <Text style={styles.changePhoto}>Tap to choose a different photo</Text>
          </Pressable>
        ) : (
          <View style={styles.photoChooser}>
            <Text style={styles.photoGlyph}>▧</Text>
            <Text style={styles.photoTitle}>Choose the moment</Text>
            <View style={styles.buttonRow}>
              <SecondaryButton label="Take photo" onPress={() => choosePhoto('camera')} />
              <SecondaryButton label="Photo library" onPress={() => choosePhoto('library')} />
            </View>
          </View>
        )}
        <Field label="What do you remember?" value={note} onChangeText={setNote} multiline placeholder="The light, the people, how it felt…" />
        <View style={styles.twoColumns}>
          <View style={styles.column}><Field label="Mood" value={mood} onChangeText={setMood} placeholder="calm" /></View>
          <View style={styles.column}><Field label="Tags" value={tags} onChangeText={setTags} placeholder="trip, summer" /></View>
        </View>
        <Field label="Place hint" value={placeHint} onChangeText={setPlaceHint} placeholder="near the old window" />
        <Pressable style={styles.locationButton} onPress={useLocation} disabled={locating}>
          <Text style={styles.locationButtonText}>{locating ? 'Finding location…' : coords ? '✓ Current location attached' : '⌖ Attach current location'}</Text>
        </Pressable>
        <PrimaryButton label={saving ? 'Saving…' : 'Save memory'} onPress={save} disabled={saving} />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function SearchScreen({ request, apiUrl, apiKey }: { request: Requester; apiUrl: string; apiKey: string }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Memory[]>([]);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);

  async function search() {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append('query', query.trim());
      form.append('top_k', '10');
      const response = (await request('/search', { method: 'POST', body: form })) as SearchResponse;
      setResults(response.matches || []);
      setSearched(true);
    } catch (error) {
      Alert.alert('Search failed', errorMessage(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={styles.fill}>
      <FlatList
        data={results}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={styles.listContent}
        keyboardShouldPersistTaps="handled"
        ListHeaderComponent={
          <>
            <Header eyebrow="FIND" title="Search your life" subtitle="Describe an image, a mood, a place, or a detail." />
            <View style={styles.searchRow}>
              <TextInput style={styles.searchInput} value={query} onChangeText={setQuery} placeholder="that quiet café…" placeholderTextColor="#9B8E81" returnKeyType="search" onSubmitEditing={search} />
              <Pressable style={styles.searchButton} onPress={search}><Text style={styles.searchButtonText}>⌕</Text></Pressable>
            </View>
            {loading ? <ActivityIndicator color="#715840" style={{ marginVertical: 30 }} /> : null}
            {searched && !loading && !results.length ? <NoteCard text="No matching memories yet. Try another detail." /> : null}
          </>
        }
        renderItem={({ item }) => <MemoryCard memory={item} apiUrl={apiUrl} apiKey={apiKey} />}
      />
    </View>
  );
}

function AskScreen({ request, apiUrl, apiKey }: { request: Requester; apiUrl: string; apiKey: string }) {
  const [message, setMessage] = useState('');
  const [answer, setAnswer] = useState('');
  const [related, setRelated] = useState<Memory[]>([]);
  const [loading, setLoading] = useState(false);

  async function ask() {
    if (!message.trim()) return;
    setLoading(true);
    try {
      const response = (await request('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message.trim(), top_k: 3 }),
      })) as ChatResponse;
      setAnswer(response.answer);
      setRelated(response.memories || []);
    } catch (error) {
      Alert.alert('Bepo could not answer', errorMessage(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        <Header eyebrow="RECALL" title="Ask Bepo" subtitle="Ask naturally. Bepo will look through what you saved." />
        <View style={styles.askCard}>
          <TextInput style={styles.askInput} value={message} onChangeText={setMessage} multiline placeholder="Where was that calm café with the cat?" placeholderTextColor="#9B8E81" />
          <PrimaryButton label={loading ? 'Remembering…' : 'Ask Bepo'} onPress={ask} disabled={loading} />
        </View>
        {answer ? (
          <View style={styles.answerCard}>
            <Text style={styles.answerEyebrow}>BEPO REMEMBERS</Text>
            <Text style={styles.answerText}>{answer}</Text>
          </View>
        ) : null}
        {related.map((item) => <MemoryCard key={item.id} memory={item} apiUrl={apiUrl} apiKey={apiKey} />)}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function SettingsScreen(props: ConnectionProps) {
  return (
    <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView contentContainerStyle={styles.scrollContent} keyboardShouldPersistTaps="handled">
        <Header eyebrow="PRIVATE BY DEFAULT" title="Connection" subtitle="Your key is stored securely on this device and is never included in the app code." />
        <ConnectionForm {...props} />
        <NoteCard title="About this Bepo" text="Your photos and memories live on your private Railway service. The phone app only displays and adds to them." />
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

type ConnectionProps = {
  url: string; setUrl: (value: string) => void; apiKey: string; setApiKey: (value: string) => void;
  error: string; loading: boolean; onSave: () => void;
};

function ConnectionForm({ url, setUrl, apiKey, setApiKey, error, loading, onSave }: ConnectionProps) {
  return (
    <View style={styles.connectionCard}>
      <Field label="Bepo server" value={url} onChangeText={setUrl} autoCapitalize="none" keyboardType="url" />
      <Field label="Private API key" value={apiKey} onChangeText={setApiKey} autoCapitalize="none" secureTextEntry placeholder="Paste BEPO_API_KEY" />
      {error ? <Text style={styles.errorText}>{error}</Text> : null}
      <PrimaryButton label={loading ? 'Checking…' : 'Connect securely'} onPress={onSave} disabled={loading} />
    </View>
  );
}

function NoteCard({ title, text }: { title?: string; text: string }) {
  return <View style={styles.noteCard}>{title ? <Text style={styles.noteCardTitle}>{title}</Text> : null}<Text style={styles.noteCardText}>{text}</Text></View>;
}

function Field({ label, multiline, ...inputProps }: { label: string; multiline?: boolean } & React.ComponentProps<typeof TextInput>) {
  return (
    <View style={styles.field}>
      <Text style={styles.fieldLabel}>{label}</Text>
      <TextInput {...inputProps} multiline={multiline} style={[styles.input, multiline && styles.multilineInput]} placeholderTextColor="#9B8E81" />
    </View>
  );
}

function PrimaryButton({ label, onPress, disabled = false }: { label: string; onPress: () => void; disabled?: boolean }) {
  return <Pressable style={[styles.primaryButton, disabled && styles.disabledButton]} onPress={onPress} disabled={disabled}><Text style={styles.primaryButtonText}>{label}</Text></Pressable>;
}

function SecondaryButton({ label, onPress }: { label: string; onPress: () => void }) {
  return <Pressable style={styles.secondaryButton} onPress={onPress}><Text style={styles.secondaryButtonText}>{label}</Text></Pressable>;
}

