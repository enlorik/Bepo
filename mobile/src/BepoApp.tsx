import { File } from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
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

type Screen = 'chat' | 'memories' | 'settings';

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
type Requester = (path: string, options?: RequestInit) => Promise<any>;

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  photoUri?: string;
  memories?: Memory[];
  meta?: string;
  error?: boolean;
};

type Coordinates = { lat: number; lon: number };

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

function messageId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function BepoApp() {
  const [screen, setScreen] = useState<Screen>('chat');
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
      setScreen('chat');
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
  onMemorySaved,
  onOpenMemories,
  onOpenSettings,
}: {
  request: Requester;
  apiUrl: string;
  apiKey: string;
  memoryCount: number;
  onMemorySaved: () => Promise<void>;
  onOpenMemories: () => void;
  onOpenSettings: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState('');
  const [pendingPhoto, setPendingPhoto] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [sending, setSending] = useState(false);
  const listRef = useRef<FlatList<ChatMessage>>(null);

  function addMessage(message: ChatMessage) {
    setMessages((current) => [...current, message]);
  }

  function updateMessage(id: string, patch: Partial<ChatMessage>) {
    setMessages((current) => current.map((message) => (message.id === id ? { ...message, ...patch } : message)));
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
        ? await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.85 })
        : await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 0.85 });
      if (!result.canceled && result.assets[0]) setPendingPhoto(result.assets[0]);
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
    const text = (textOverride ?? draft).trim();
    const photo = pendingPhoto;
    if (sending || (!text && !photo)) return;

    const userId = messageId('user');
    addMessage({
      id: userId,
      role: 'user',
      text: text || 'Remember this.',
      photoUri: photo?.uri,
      meta: photo ? 'Saving memory…' : undefined,
    });
    if (textOverride === undefined) setDraft('');
    setPendingPhoto(null);
    setSending(true);

    try {
      if (photo) {
        const coordinates = await getAutomaticLocation();
        const form = new FormData();
        const photoFile = new File(photo.uri);
        form.append('photo', photoFile, photo.fileName || photoFile.name || `bepo-${Date.now()}.jpg`);
        if (text) form.append('note', text);
        if (coordinates) {
          form.append('lat', String(coordinates.lat));
          form.append('lon', String(coordinates.lon));
        }
        await request('/memory', { method: 'POST', body: form });
        updateMessage(userId, { meta: coordinates ? 'Saved with location' : 'Saved' });
        addMessage({
          id: messageId('bepo'),
          role: 'assistant',
          text: coordinates
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

  const canSend = Boolean(draft.trim() || pendingPhoto) && !sending;

  return (
    <View style={styles.fill}>
      <ChatHeader memoryCount={memoryCount} onOpenMemories={onOpenMemories} onOpenSettings={onOpenSettings} />
      <KeyboardAvoidingView style={styles.fill} behavior={Platform.OS === 'ios' ? 'padding' : undefined} keyboardVerticalOffset={0}>
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
              <Image source={{ uri: pendingPhoto.uri }} style={styles.attachmentImage} />
              <View style={styles.attachmentCopy}>
                <Text style={styles.attachmentTitle}>New memory</Text>
                <Text style={styles.attachmentMeta}>Time and location added automatically</Text>
              </View>
              <Pressable accessibilityLabel="Remove attached photo" style={styles.removeAttachment} onPress={() => setPendingPhoto(null)}>
                <Text style={styles.removeAttachmentText}>×</Text>
              </Pressable>
            </View>
          ) : null}
          <View style={styles.composer}>
            <View style={styles.composerTools}>
              <Pressable accessibilityLabel="Take a photo" style={styles.composerToolButton} onPress={() => choosePhoto('camera')}>
                <Text style={styles.composerToolGlyph}>◎</Text>
                <Text style={styles.composerToolLabel}>Camera</Text>
              </Pressable>
              <Pressable accessibilityLabel="Choose from photos" style={styles.composerToolButton} onPress={() => choosePhoto('library')}>
                <Text style={styles.composerToolGlyph}>▧</Text>
                <Text style={styles.composerToolLabel}>Photos</Text>
              </Pressable>
            </View>
            <TextInput
              style={styles.composerInput}
              value={draft}
              onChangeText={setDraft}
              multiline
              placeholder={pendingPhoto ? 'Add what you remember…' : 'Ask Bepo anything…'}
              placeholderTextColor="#8B8B85"
            />
            <Pressable
              accessibilityLabel="Send message"
              style={[styles.sendButton, !canSend && styles.sendButtonDisabled]}
              onPress={() => sendMessage()}
              disabled={!canSend}
            >
              <Text style={styles.sendButtonText}>↑</Text>
            </Pressable>
          </View>
          <Text style={styles.composerHint}>
            {pendingPhoto ? 'This photo will become a memory.' : 'A photo becomes a memory. Text alone asks a question.'}
          </Text>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

function ChatHeader({ memoryCount, onOpenMemories, onOpenSettings }: {
  memoryCount: number;
  onOpenMemories: () => void;
  onOpenSettings: () => void;
}) {
  return (
    <View style={styles.chatHeader}>
      <View style={styles.headerIdentity}>
        <BrandMark size={38} />
        <View>
          <Text style={styles.chatHeaderTitle}>Bepo</Text>
          <Text style={styles.chatHeaderSubtitle}>Your private memory companion</Text>
        </View>
      </View>
      <View style={styles.headerActions}>
        <Pressable accessibilityLabel="Open memories" style={styles.headerButton} onPress={onOpenMemories}>
          <Text style={styles.headerButtonText}>▦</Text>
          {memoryCount ? <View style={styles.memoryBadge}><Text style={styles.memoryBadgeText}>{memoryCount > 99 ? '99+' : memoryCount}</Text></View> : null}
        </Pressable>
        <Pressable accessibilityLabel="Open settings" style={styles.headerButton} onPress={onOpenSettings}>
          <Text style={styles.headerButtonText}>•••</Text>
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

function MemoriesScreen({ memories, apiUrl, apiKey, loading, refreshing, onRefresh, onBack }: {
  memories: Memory[];
  apiUrl: string;
  apiKey: string;
  loading: boolean;
  refreshing: boolean;
  onRefresh: () => void;
  onBack: () => void;
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
        renderItem={({ item }) => <MemoryCard memory={item} apiUrl={apiUrl} apiKey={apiKey} />}
      />
    </View>
  );
}

function MemoryCard({ memory, apiUrl, apiKey, compact = false }: {
  memory: Memory;
  apiUrl: string;
  apiKey: string;
  compact?: boolean;
}) {
  const description = memory.bepo_summary || memory.user_note || memory.note || 'A saved moment';
  return (
    <View style={[styles.memoryCard, compact && styles.inlineMemoryCard]}>
      <Image
        source={{ uri: absoluteUrl(apiUrl, memory.image_url), headers: { 'X-API-Key': apiKey } }}
        style={[styles.memoryImage, compact && styles.inlineMemoryImage]}
        resizeMode="cover"
      />
      <View style={styles.memoryBody}>
        <View style={styles.memoryMetaRow}>
          <Text style={styles.memoryDate}>{formatDate(memory.timestamp)}</Text>
          {memory.mood ? <Text style={styles.pill}>{memory.mood}</Text> : null}
        </View>
        <Text style={[styles.memoryDescription, compact && styles.inlineMemoryDescription]}>{description}</Text>
        {memory.tags ? <Text style={styles.memoryTags}>{memory.tags}</Text> : null}
        {memory.place_hint ? <Text style={styles.memoryPlace}>⌖ {memory.place_hint}</Text> : null}
        {memory.map_url ? (
          <Pressable onPress={() => Linking.openURL(memory.map_url!)}>
            <Text style={styles.mapLink}>Open location ↗</Text>
          </Pressable>
        ) : null}
      </View>
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
            title="Automatic location"
            text="When you send a photo, Bepo adds your current location automatically if you allow location access. A photo can still be saved when location is unavailable."
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

