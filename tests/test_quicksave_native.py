#!/usr/bin/env python3
"""Test checkpoint control, real save writes, mount exclusion and split keyboard routing."""
import os
from pathlib import Path
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sdk = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip()

CASES = {
    'control': r'''
#include <assert.h>
#include "runtime_quicksave.c"
static int called, operation, allow=1;
int runtime_savedata_quick_action(int load) { ++called; operation=load; return allow; }
void runtime_savedata_dir(char *out,size_t size) { snprintf(out,size,"/unused"); }
const char *runtime_file_user_dir(void) { return "/unused"; }
void runtime_restart(void) { assert(0); }
int runtime_memory_is_mapped(uintptr_t p,uint64_t n) { (void)p;(void)n;return 0; }
int main(void) {
    enabled=1;
    runtime_quicksave_keys(0,0,0);
    runtime_quicksave_keys(1,0,0); assert(called==1 && !operation && !pending);
    for (int i=0;i<100;++i) runtime_quicksave_keys(1,0,0);
    assert(called==1); /* held X is one action */
    runtime_quicksave_keys(0,0,0);
    runtime_quicksave_keys(0,1,1); /* text/menu or modifier held */
    runtime_quicksave_keys(0,1,0); assert(called==1);
    runtime_quicksave_keys(0,0,0);
    runtime_quicksave_keys(0,1,0); assert(called==2 && operation);
    runtime_quicksave_keys(0,0,0);
    runtime_quicksave_keys(1,1,0); assert(called==2); /* ambiguous chord */
    runtime_quicksave_keys(0,0,0);
    allow=0;
    runtime_quicksave_keys(1,0,0); assert(pending==1);
    allow=1;
    runtime_quicksave_keys(0,0,0); assert(!pending); /* retry after writes close */
    auto_confirm=1;
    assert(runtime_quicksave_autoconfirm()==0);
    runtime_quicksave_request(1); assert(!pending);
    runtime_quicksave_loaded(); assert(runtime_quicksave_autoconfirm()==-1);
    allow=0;
    runtime_quicksave_keys(0,0,0);
    runtime_quicksave_keys(1,0,0); assert(pending==1);
    requested_at=clock_ms()-16000;
    runtime_quicksave_keys(0,0,0); assert(!pending);
    puts("PASS: X/M edge triggering, held keys, modifiers/text capture, deferred save, autoload release");
}
''',
    'freshness': r'''
#include <assert.h>
#include "runtime_quicksave.c"
static unsigned char image[0x5578a40], manager[0x1579];
static int requests;
int runtime_savedata_quick_action(int load) { (void)load;return 0; }
void runtime_savedata_dir(char *out,size_t size) { snprintf(out,size,"/unused"); }
const char *runtime_file_user_dir(void) { return "/unused"; }
void runtime_restart(void) { assert(0); }
int runtime_memory_is_mapped(uintptr_t p,uint64_t n) {
    return p==(uintptr_t)manager && n<=sizeof(manager);
}
static ABI void save_request(void) { ++requests; }
int main(void) {
    enabled=1;
    runtime_quicksave_bind(image,10); assert(!game_save);
    runtime_quicksave_bind(image,sizeof(image)); assert(!game_save);
    assert(runtime_quicksave_execute(0)==-1); /* unsupported builds fail closed */
    guest_image=image; game_save=save_request;
    assert(runtime_quicksave_execute(0)==0 && !requests); /* menu/not ready */
    uintptr_t context=1, ptr=(uintptr_t)manager;
    memcpy(image+0x553b0c8,&context,sizeof(context));
    memcpy(image+0x5556678,&ptr,sizeof(ptr));
    manager[0x1578]=1;
    /* Use a successful stand-in helper; only the freshness gate is under test. */
    setenv("BB_QUICKSAVE_PYTHON","/usr/bin/true",1);
    setenv("BB_QUICKSAVE_HELPER","/unused",1);
    for (int i=0;i<2;++i) {
        fresh_requested=0;
        runtime_quicksave_committed("/savedata0/userdata0000");
        runtime_quicksave_committed("/savedata0/userdata0010");
        assert(runtime_quicksave_execute(0)==0 && requests==i+1);
        assert(runtime_quicksave_execute(0)==0); /* old writes never qualify */
        runtime_quicksave_committed("/savedata0/userdata0010");
        assert(runtime_quicksave_execute(0)==0); /* profile alone is insufficient */
        runtime_quicksave_committed("/savedata0/userdata0000.bbtmp");
        assert(runtime_quicksave_execute(0)==0);
        runtime_quicksave_committed("/savedata0/userdata0007");
        assert(runtime_quicksave_execute(0)==1); /* supports any character slot */
    }
    fresh_requested=0;
    assert(!runtime_quicksave_execute(0));
    runtime_quicksave_committed("/savedata0/userdata0000");
    assert(!runtime_quicksave_execute(0)); /* wait between the two save mounts */
    runtime_quicksave_committed("/savedata0/userdata0010");
    assert(runtime_quicksave_execute(0)==1);
    puts("PASS: fresh game request, two successive saves, both commits required, unsupported builds rejected");
}
''',
    'mounts': r'''
#define _GNU_SOURCE
#include <assert.h>
#include "runtime_savedata.c"
static int calls, idle=1;
int runtime_quicksave_execute(int load) { ++calls; return load ? -1 : 1; }
void runtime_quicksave_init(const char *title) { (void)title; }
void runtime_quicksave_loaded(void) {}
int runtime_file_savedata_idle(void) { return idle; }
int runtime_file_mount(const char *g,const char *h) { (void)g;(void)h;return 0; }
void runtime_file_unmount(const char *p) { (void)p; }
const char *runtime_file_user_dir(void) { return "/unused"; }
uintptr_t runtime_lookup(const RuntimeExport *e,size_t n,const char *s) { (void)e;(void)n;(void)s;return 0; }
int main(void) {
    slots[5].used=1;
    assert(!runtime_savedata_quick_action(0) && !calls);
    slots[5].used=0; idle=0;
    assert(!runtime_savedata_quick_action(0) && !calls);
    idle=1;
    pthread_mutex_lock(&lock);
    assert(!runtime_savedata_quick_action(0) && !calls);
    pthread_mutex_unlock(&lock);
    assert(runtime_savedata_quick_action(0)==1 && calls==1);
    assert(runtime_savedata_quick_action(1)==-1 && calls==2);
    assert(!pthread_mutex_trylock(&lock)); pthread_mutex_unlock(&lock);
    puts("PASS: mounted saves/open files exclude snapshots; busy lock defers; errors release lock");
}
''',
    'bindings': r'''
#define _GNU_SOURCE
#include <assert.h>
#include "runtime_pad.c"
static int seen_save, seen_load, seen_blocked, captured;
static bool host_keys[SDL_SCANCODE_COUNT];
void runtime_quicksave_keys(int s,int l,int b) { seen_save=s; seen_load=l; seen_blocked=b; }
int runtime_quicksave_autoconfirm(void) { return -1; }
int bbgpu_overlay_captures_input(void) { return captured; }
const bool *bbgpu_split_keyboard(void) { return host_keys; }
uintptr_t runtime_lookup(const RuntimeExport *e,size_t n,const char *s) { (void)e;(void)n;(void)s;return 0; }
int main(void) {
    sdl_ready=-1;
    PadData data;
    host_keys[SDL_SCANCODE_X]=true;
    sample_host(&data); assert(seen_save && !seen_load && !seen_blocked && !data.buttons);
    host_keys[SDL_SCANCODE_X]=false; host_keys[SDL_SCANCODE_M]=true;
    sample_host(&data); assert(!seen_save && seen_load && !seen_blocked && !data.buttons);
    captured=1; sample_host(&data); assert(seen_blocked);
    captured=0; host_keys[SDL_SCANCODE_LGUI]=true;
    sample_host(&data); assert(seen_blocked);
    host_keys[SDL_SCANCODE_LGUI]=false; host_keys[SDL_SCANCODE_LSHIFT]=true;
    sample_host(&data); assert(seen_blocked);
    host_keys[SDL_SCANCODE_LSHIFT]=false; host_keys[SDL_SCANCODE_M]=false;
    host_keys[SDL_SCANCODE_S]=true; sample_host(&data);
    assert(!seen_save && !seen_load && data.left_y==255);
    puts("PASS: fixed X/M, gameplay mappings preserved, modifier and text/settings capture");
}
''',
    'files': r'''
#include "runtime_file.c"
#include <assert.h>
static int commits;
static int32_t guest_errno;
int runtime_quicksave_enabled(void) { return 1; }
void runtime_quicksave_committed(const char *path) { assert(strstr(path,"userdata")); ++commits; }
int32_t *runtime_errno(void) { return &guest_errno; }
int32_t runtime_guest_errno(int e) { return e; }
uintptr_t runtime_lookup(const RuntimeExport *e,size_t n,const char *s) { (void)e;(void)n;(void)s;return 0; }
int main(void) {
    char root[]="/tmp/bb-quicksave-files-XXXXXX";
    assert(mkdtemp(root));
    assert(!runtime_file_mount("/savedata0",root));
    const char *path="/savedata0/userdata0000";
    assert(runtime_file_savedata_idle());
    int fd=(int)do_open(path,0x602,0600); assert(fd>=3);
    assert(!runtime_file_savedata_idle());
    assert(do_write(fd,"hello",5)==5);
    assert(!commits && !do_close(fd) && commits==1 && runtime_file_savedata_idle());
    fd=(int)do_open(path,0,0); assert(fd>=3);
    assert(!do_close(fd) && commits==1); /* reads never qualify */
    fd=(int)do_open(path,2,0); assert(fd>=3);
    assert(!do_close(fd) && commits==1); /* writable open without writes never qualifies */
    fd=(int)do_open(path,2,0); assert(fd>=3);
    assert(do_write(fd,"bad",3)==3);
    assert(do_pwrite(fd,"bad",3,-1)<0);
    assert(!do_close(fd) && commits==1); /* failed writes never qualify */
    fd=(int)do_open(path,2,0); assert(fd>=3);
    assert(do_pwrite(fd,"world",5,0)==5);
    assert(!do_close(fd) && commits==2);
    assert(!path_op(path,2,0));
    assert(!rmdir(root));
    puts("PASS: only completed writes count, read-only/unwritten/failed writes excluded");
}
''',
}

with tempfile.TemporaryDirectory(prefix='bb-quicksave-native-') as temp:
    for name, source in CASES.items():
        c = Path(temp) / (name + '.c')
        exe = Path(temp) / name
        c.write_text(source)
        command = ['/usr/bin/clang', '-arch', 'x86_64', '-isysroot', sdk, '-std=c11', '-O1',
                   '-Wall', '-Wextra', '-Werror', '-pthread', '-I' + str(ROOT),
                   '-I' + str(ROOT / 'src'), str(c), '-o', str(exe)]
        if name == 'bindings':
            deps = ROOT / 'deps/x86_64'
            command += [str(ROOT / 'src/runtime_host.c'), '-I' + str(deps / 'include'), '-L' + str(deps / 'lib'),
                        '-Wl,-rpath,' + str(deps / 'lib'), '-lSDL3']
        subprocess.run(command, check=True)
        subprocess.run([str(exe)], check=True)
