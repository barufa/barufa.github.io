# DEV.to (and any other Forem instance) proxies every inline image through
# Cloudflare Images with format=auto. That proxy does not rasterize SVG: it
# returns the SVG bytes untouched while still advertising
# `Content-Type: image/webp`, so the browser fails to decode it and the image
# renders as broken. This breaks posts imported into DEV.to via RSS.
#
# The site keeps serving the vector version; only the feeds are rewritten to
# point at a pre-rendered PNG sitting next to each SVG. Generate it with:
#
#   rsvg-convert -w 1560 -b white assets/img/foo.svg -o assets/img/foo.png
#
# The white background is not optional: SVGs here use `fill: none` plus a
# `prefers-color-scheme: dark` block, neither of which survives rasterization.
module Jekyll
  module FeedSvgToPng
    IMG_SRC = %r!(<img\b[^>]*?\bsrc\s*=\s*")([^"]+\.svg)(")!i.freeze
    MEDIA_URL = %r!(<media:content\b[^>]*?\burl\s*=\s*")([^"]+\.svg)(")!i.freeze

    def self.rewrite!(site, page)
      [IMG_SRC, MEDIA_URL].each do |pattern|
        page.output = page.output.gsub(pattern) do
          prefix = Regexp.last_match(1)
          url = Regexp.last_match(2)
          suffix = Regexp.last_match(3)
          "#{prefix}#{png_for(site, url) || url}#{suffix}"
        end
      end
    end

    # Returns the URL of the PNG twin of `url`, or nil when it does not exist
    # on disk. Missing twins are left alone so a build never silently ships a
    # dead image reference.
    def self.png_for(site, url)
      candidate = url.sub(%r!\.svg\z!i, '.png')
      return candidate if File.exist?(site.in_source_dir(source_path(site, candidate)))

      Jekyll.logger.warn 'Feed:', "no PNG twin for #{url}; DEV.to will not render it"
      nil
    end

    # Maps a feed URL (absolute or root-relative) back to a path inside the
    # site source, so the PNG can be checked for existence.
    def self.source_path(site, url)
      path = url.sub(%r!\Ahttps?://[^/]+!, '')
      baseurl = site.baseurl.to_s
      path = path.sub(%r!\A#{Regexp.escape(baseurl)}!, '') unless baseurl.empty?
      path.sub(%r!\A/!, '')
    end

    def self.feed?(page)
      page.output_ext == '.xml' &&
        (page.output.include?('<feed') || page.output.include?('<rss'))
    end
  end
end

Jekyll::Hooks.register :pages, :post_render do |page|
  Jekyll::FeedSvgToPng.rewrite!(page.site, page) if Jekyll::FeedSvgToPng.feed?(page)
end
