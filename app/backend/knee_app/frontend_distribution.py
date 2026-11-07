"""Opt-in public build delivery; it never mounts a directory or falls back on API paths."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from starlette.requests import Request
from starlette.responses import Response

from .errors import ApiError


NO_CACHE='no-cache, max-age=0, must-revalidate'
IMMUTABLE='public, max-age=31536000, immutable'
MAX_FILE_BYTES=16*1024*1024
MAX_BUILD_BYTES=64*1024*1024
ROOT_FILES={
    '/index.html':'text/html',
    '/sw.js':'application/javascript',
    '/manifest.webmanifest':'application/manifest+json',
    '/pwa-version.json':'application/json',
    '/icons/kneeoai.svg':'image/svg+xml',
    '/icons/kneeoai-maskable.svg':'image/svg+xml',
    '/icons/kneeoai-192.png':'image/png',
    '/icons/kneeoai-512.png':'image/png',
}
ASSET_TYPES={
    'js':'application/javascript','css':'text/css','png':'image/png',
    'svg':'image/svg+xml','jpg':'image/jpeg','jpeg':'image/jpeg',
    'webp':'image/webp','ico':'image/x-icon','woff':'font/woff','woff2':'font/woff2',
}
HASHED_ASSET=re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]*-[A-Za-z0-9_-]{8,}\.(js|css|png|svg|jpg|jpeg|webp|ico|woff|woff2)$')
VUE_DETAIL=re.compile(r'^/evaluaciones/[A-Za-z0-9][A-Za-z0-9._:-]{0,127}/(?:editar|radiografia|revision|resumen)$')
VUE_PAGES={'/','/acceso','/evaluaciones','/evaluaciones/nueva'}


def is_link(path):
    return path.is_symlink() or getattr(path,'is_junction',lambda:False)()


def plain_path(path):
    """Reject links before resolving, including a linked build root/ancestor on Windows."""
    path=Path(path).absolute()
    if any(is_link(candidate) for candidate in (path,*path.parents)):
        raise ValueError('Frontend distribution must not contain symbolic links or junctions')
    return path.resolve()


@dataclass(frozen=True)
class PublicFile:
    path:Path
    content:bytes
    media_type:str
    cache_control:str
    etag:str


class FrontendDistribution:
    def __init__(self,path,*,database_path,private_files_path):
        self.root=plain_path(path)
        private_root=Path(private_files_path).resolve()
        database=Path(database_path).resolve()
        if not self.root.is_dir() or self.root.is_relative_to(private_root) or private_root.is_relative_to(self.root) or database.is_relative_to(self.root):
            raise ValueError('Frontend distribution must be a separate public build directory')
        self.files={}
        self.total_bytes=0
        for url,media_type in ROOT_FILES.items():
            self._add(url,media_type,NO_CACHE,required=url=='/index.html')
        assets=self.root/'assets'
        if assets.is_dir() and not is_link(assets):
            for item in sorted(assets.iterdir()):
                matched=HASHED_ASSET.fullmatch(item.name)
                if matched:
                    self._add('/assets/'+item.name,ASSET_TYPES[matched.group(1)],IMMUTABLE)

    def _add(self,url,media_type,cache_control,*,required=False):
        candidate=self.root/url.lstrip('/')
        try:
            path=plain_path(candidate)
            if not path.is_relative_to(self.root) or not path.is_file():
                raise ValueError('Not a public build file')
            with path.open('rb') as stream:
                content=stream.read(MAX_FILE_BYTES+1)
            if len(content)>MAX_FILE_BYTES or self.total_bytes+len(content)>MAX_BUILD_BYTES:
                raise ValueError('Frontend distribution exceeds its technical size limit')
        except (OSError,ValueError):
            if required:
                raise ValueError('Frontend distribution requires a regular index.html') from None
            return
        self.total_bytes+=len(content)
        self.files[url]=PublicFile(path,content,media_type,cache_control,'"'+hashlib.sha256(content).hexdigest()+'"')

    def entry_for(self,path):
        if path in self.files:
            return self.files[path]
        canonical=path[:-1] if path.endswith('/') and path!='/' else path
        if canonical in VUE_PAGES or VUE_DETAIL.fullmatch(canonical):
            return self.files['/index.html']
        return None

    def response(self,request,entry):
        # Content and inventory are immutable for this app instance. Rebuild/restart for a release.
        try:
            live_path=plain_path(entry.path)
            if not live_path.is_relative_to(self.root) or not live_path.is_file():
                raise ValueError('Public file is unavailable')
        except (OSError,ValueError):
            raise ApiError(404,'RESOURCE_NOT_FOUND','No se encontró el recurso solicitado.')
        headers={'Cache-Control':entry.cache_control,'ETag':entry.etag,'X-Content-Type-Options':'nosniff'}
        if request.url.path=='/sw.js':
            headers['Service-Worker-Allowed']='/'
        if request.headers.get('if-none-match')==entry.etag:
            return Response(status_code=304,headers=headers)
        headers['Content-Length']=str(len(entry.content))
        return Response(entry.content if request.method=='GET' else b'',media_type=entry.media_type,headers=headers)


class FrontendDistributionMiddleware:
    def __init__(self,app,distribution):
        self.app,self.distribution=app,distribution

    async def __call__(self,scope,receive,send):
        path=scope.get('path','')
        if scope['type']!='http' or scope.get('method') not in {'GET','HEAD'} or path=='/api' or path.startswith('/api/'):
            return await self.app(scope,receive,send)
        entry=self.distribution.entry_for(path)
        if entry is None:
            return await self.app(scope,receive,send)
        raw=scope.get('raw_path',path.encode()).split(b'?',1)[0]
        if b'%' in raw or b'\\' in raw or b'\x00' in raw:
            raise ApiError(404,'RESOURCE_NOT_FOUND','No se encontró el recurso solicitado.')
        request=Request(scope,receive)
        response=self.distribution.response(request,entry)
        scope.setdefault('state',{})['public_frontend']=True
        await response(scope,receive,send)
